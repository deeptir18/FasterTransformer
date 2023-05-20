# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import grpc
import argparse
import collections
import configparser
import os
from re import I
import sys
import math
import logging
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
import csv
import os
import struct
import multiprocessing as mp
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path + "/../../../3rdparty/transformers/src/")

from transformers import PreTrainedTokenizerFast
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path + os.environ['PROTO_PATH'])
sys.path.append(dir_path + "/../../..")
## TODO: is there a way to do this so this path isn't hardcoded into the submodule
sys.path.append(os.environ['PROTO_PATH'])
sys.path.append(os.path.join(os.environ['PROTO_PATH'], "proto_out"))
print(sys.path)
from proto_out import lm_retriever_pb2, lm_retriever_pb2_grpc # transformers-4.10.0-py3
from examples.pytorch.t5.utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder
from examples.pytorch.t5.utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from examples.pytorch.decoding.utils.recover_bpe import recover_bpe

LOGGER = logging.getLogger(__name__)

class LmRetrieverServicer(lm_retriever_pb2_grpc.LmRetrieverServicer):
    def __init__(self, args_dict):
        super().__init__()
        self.args_dict = args_dict
        args_dict['beam_size'] = 1
        args_dict['topk'] = 1
        args_dict['topp'] = 0.0
        args_dict['temperature'] = 1.0
        args_dict['len_penalty'] = 0.0
        args_dict['beam_search_diversity_rate'] = 0
        args_dict['repetition_penalty'] = 1.0
        self.fifo_path = args_dict['fifo_path']

        model_path = args_dict['model_path'] if args_dict['model_path'] != None else args_dict['model']
        self.ft_t5 = init_ftt5(args_dict)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        try:
            self.fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        except:
            self.fast_tokenizer = T5Tokenizer.from_pretrained(model_path)
        ## TODO: open grpc client connection to the Rust tokenizer for sending
        ## requests to the next service
    
    def get(self, var):
        return self.args_dict[var]
    
    def RunLanguageModel(self, request, context):
        batch_size = len(request.query)
        LOGGER.info(f"Received non pipelined request with batch size {batch_size}")
        input_texts = [f"{query.prepend}: {query.prompt}" if
                       query.prepend != None else query.prompt for query in request.query]
        if len(input_texts) != self.get('batch_size'):
            LOGGER.warn(f"Got input text length {len(input_texts)}; batch size"\
                        f"is {self.get('batch_size')}.")
            ## TODO: add Error option to RPC
            return lm_retriever_pb2.Empty()
        input_tokens = self.tokenizer(input_texts, 
                                      return_tensors = 'pt', 
                                      padding = True)
        bad_words_list = None
        stop_words_list = None
        tmp_beam_size =  self.get('beam_size')
        ft_decoding_outputs, ft_decoding_seq_lens = self.ft_t5(input_tokens,
                                                                  None,
                                                                  tmp_beam_size,
                                                                  self.get('max_seq_len'),
                                                                  self.get('topk'),
                                                                  self.get('topp'),
                                                               beam_search_diversity_rate=self.get('beam_search_diversity_rate'),
                                                                  is_return_output_log_probs=False,
                                                                  is_return_cum_log_probs=False,
                                                                  repetition_penalty=self.get('repetition_penalty'),
                                                                  temperature=self.get('temperature'),
                                                                  len_penalty=self.get('len_penalty'),
                                                                  bad_words_list=bad_words_list,
                                                                  stop_words_list=stop_words_list,
                                                                  intermediate_fifo_file=None)
        ## tokenize the outputs and construct a token vec batch
        lm_output_batch = lm_retriever_pb2.LmOutputBatch(outputs=[])
        for b in range(self.get('batch_size')):
            seq_len = ft_decoding_seq_lens[b][0]
            token_list = self.fast_tokenizer.decode(
                    ft_decoding_outputs[b][0][:seq_len],
                    skip_special_tokens = True)
            print(token_list)
            lm_output = lm_retriever_pb2.LmOutput(output = token_list,
                                                  request_id=request.query[b].request_id)
            lm_output_batch.outputs.append(lm_output)
        ## TODO: send token_vec_batch to the next microservice
        return lm_retriever_pb2.Empty()
    
    def RunLanguageModelPipelined(self, request, context):
        batch_size = len(request.query)
        print(f"Received pipelined request with batch_size {batch_size}")
        input_texts = [f"{query.prepend}: {query.prompt}" if
                       query.prepend != None else query.prompt for query in request.query]
        input_tokens = self.tokenizer(input_texts, return_tensors = 'pt', padding =True)
        if not(os.path.exists(self.fifo_path)):
            os.mkfifo(self.fifo_path, 0x600)
        p = mp.Process(target = read_from_pipe_and_tokenize,
                       args = (
                           self.fast_tokenizer,
                           self.fifo_path,
                           batch_size,
                           request.query[0].request_id,
                           ))
        p.start()
        bad_words_list = None
        stop_words_list = None
        tmp_beam_size =  self.get('beam_size')
        ft_decoding_outputs, ft_decoding_seq_lens = self.ft_t5(input_tokens,
                                                                  None,
                                                                  tmp_beam_size,
                                                                  self.get('max_seq_len'),
                                                                  self.get('topk'),
                                                                  self.get('topp'),
                                                               beam_search_diversity_rate=self.get('beam_search_diversity_rate'),
                                                                  is_return_output_log_probs=False,
                                                                  is_return_cum_log_probs=False,
                                                                  repetition_penalty=self.get('repetition_penalty'),
                                                                  temperature=self.get('temperature'),
                                                                  len_penalty=self.get('len_penalty'),
                                                                  bad_words_list=bad_words_list,
                                                                  stop_words_list=stop_words_list,
                                                                  intermediate_fifo_file=self.fifo_path)
        p.join()
        LOGGER.info("Joined subprocess")
        return lm_retriever_pb2.Empty()

def read_from_pipe_and_tokenize(fast_tokenizer, 
                                fifo_path, 
                                batch_size,
                                batch_start):
    LOGGER.info("In read from pipe with tokenize")
    LOGGER.info(f"fifo_path: {fifo_path}, batch_size: {batch_size}, batch_start: {batch_start}")
    r = os.open(fifo_path, os.O_RDONLY)
    tokens_read = 0
    finished = [0 for _ in range(batch_size)]
    token_batch = lm_retriever_pb2.TokenBatch(tokens=[])
    while sum(finished) < batch_size:
        index = read_int32_from_file(r)
        value = read_int32_from_file(r)
        index_done = read_bool_from_file(r)
        tokenized = fast_tokenizer.decode([value],
                                          skip_special_tokens = True)
        LOGGER.info(f"Read out of pipe: index {index}, value {value},"\
                    f"tokenized {tokenized}, is_done {index_done}")
        token = lm_retriever_pb2.Token(word=tokenized,
                                       is_start=index==0,
                                       is_end=index_done,
                                       request_id=batch_start+index)
        token_batch.tokens.append(token)
        finished[index] = index_done
    print(token_batch)
    LOGGER.info("Returning")

def init_ftt5(args_dict):
    torch.set_printoptions(precision=6)
    batch_size = args_dict['batch_size']
    beam_size = 1
    max_seq_len = args_dict['max_seq_len']
    beam_search_diversity_rate = 0
    topk = 1
    topp = 0.0
    tensor_para_size = 1
    pipeline_para_size = 1
    max_ite = args_dict['max_iteration']
    repetition_penalty = 1.0
    temperature = 1.0
    len_penalty = 0.0
    t5_with_bias = False
    use_gated_activation = False
    t5_with_moe = False
    position_embedding_type = 0
    weight_data_type = np.float32
    ## only huggingface model path supported
    model_path = args_dict['model_path'] if args_dict['model_path'] != None else args_dict['model']
    ckpt_path = args_dict['ckpt_path']
    model_type = "HuggingFace"
    load_data_type = args_dict['load_data_type']
    ## read checkpoint config if exists
    ckpt_config = configparser.ConfigParser()
    activation_type = "relu"
    LOGGER.info("\n=============== Argument ===============")
    for key in args_dict:
        LOGGER.info("{}: {}".format(key, args_dict[key]))
    LOGGER.info("========================================")

    lib_path = args_dict['lib_path']
    t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
    if dist.is_mpi_available():
        try:
            dist.init_process_group(backend='mpi')
            rank = dist.get_rank()
        except:
            rank = dist.get_rank()
    else:
        rank = 0
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    try:
        fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    except:
        fast_tokenizer = T5Tokenizer.from_pretrained(model_path)
    encoder_config = t5_model.encoder.config
    decoder_config = t5_model.decoder.config
    encoder_config.update({"num_experts": 0})
    decoder_config.update({"num_experts": 0})
    encoder_config.update({"moe_layer_index": []})
    decoder_config.update({"moe_layer_index": []})
    activation_type = encoder_config.feed_forward_proj
    if activation_type == "gated-gelu" or activation_type == "gated-relu":
        use_gated_activation = True
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1660
    # if tie_word_embeddings == True, scale the decoder output by sequence_output = sequence_output * (self.model_dim**-0.5)
    tie_word_embeddings = decoder_config.tie_word_embeddings

    q_scaling = 1.0 / (math.sqrt(encoder_config.d_kv))
    ft_encoder_weight = FTT5EncoderWeight(
        encoder_config,
        tensor_para_size,
        pipeline_para_size,
        t5_with_bias=t5_with_bias,
        use_gated_activation=use_gated_activation,
        t5_with_moe=t5_with_moe,
        position_embedding_type=position_embedding_type,
        weight_data_type=weight_data_type,
    )
    ft_decoding_weight = FTT5DecodingWeight(
        decoder_config,
        tensor_para_size,
        pipeline_para_size,
        t5_with_bias=t5_with_bias,
        use_gated_activation=use_gated_activation,
        t5_with_moe=t5_with_moe,
        position_embedding_type=position_embedding_type,
        weight_data_type=weight_data_type,
    )

    if args_dict["ckpt_path"] is not None:
        ft_encoder_weight.load_from_bin(args_dict["ckpt_path"], model_type, load_data_type)
        ft_decoding_weight.load_from_bin(args_dict["ckpt_path"], model_type, load_data_type)
    else:
        ft_encoder_weight.load_from_model(t5_model)
        ft_decoding_weight.load_from_model(t5_model)
    remove_padding = True if batch_size > 32 else False
    ft_encoder = FTT5Encoder(ft_encoder_weight.w, lib_path, encoder_config.num_heads,
                            encoder_config.d_kv, encoder_config.d_ff,
                            encoder_config.d_model, remove_padding, encoder_config.num_layers,
                            encoder_config.relative_attention_num_buckets, encoder_config.num_experts, encoder_config.moe_layer_index,
                            128, False, q_scaling, tensor_para_size, pipeline_para_size, t5_with_bias,
                            position_embedding_type, moe_k=0,
                            activation_type=activation_type,)
    ft_decoding = FTT5Decoding(ft_decoding_weight.w, lib_path,
                            decoder_config.num_heads, decoder_config.d_kv,
                            decoder_config.d_ff, encoder_config.d_model,
                            decoder_config.d_model, decoder_config.num_layers,
                            decoder_config.decoder_start_token_id, decoder_config.eos_token_id,
                            decoder_config.vocab_size,
                            q_scaling,
                            decoder_config.relative_attention_num_buckets, decoder_config.num_experts, decoder_config.moe_layer_index, max_distance=128,
                            tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size,
                            t5_with_bias=t5_with_bias,
                            position_embedding_type=position_embedding_type, moe_k=0,
                            activation_type=activation_type, tie_word_embeddings=tie_word_embeddings,)
    ft_t5 = FTT5(ft_encoder, ft_decoding)
    return ft_t5

def read_int32_from_file(f):
    buffer_size = 4
    buffer = b''
    while True:
        data = os.read(f, buffer_size)
        buffer += data
        if len(buffer) == buffer_size:
            return struct.unpack("i", buffer)[0]

def read_bool_from_file(f):
    buffer_size = 1
    buffer = b''
    while True:
        data = os.read(f, buffer_size)
        buffer += data
        if len(buffer) == buffer_size:
            if struct.unpack("?", buffer)[0]:
                return 1
            return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-pipelining', '--use_pipelining', action =
                        'store_true')
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-s', '--max_seq_len', type=int, default=128, metavar='NUMBER',
                        help='max sequence length (default: 200)')
    parser.add_argument('-beam', '--beam_width', type=int, default=1, metavar='NUMBER',
                        help='beam width (default: 4)')
    parser.add_argument("--source", default="../examples/pytorch/decoding/utils/translation/test.en",
                        help="Path to the source file.")
    parser.add_argument("--target", default="../examples/pytorch/decoding/utils/translation/test.de",
                        help="Path to the target file.")
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type for inference (default: fp32)', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('-ld', '--load_data_type', type=str, default="fp32", metavar='STRING',
                        help='data type for loading weights (default: fp32)', choices=['fp32', 'fp16'])
    parser.add_argument('-lib_path', '--lib_path', type=str, default="lib/libth_transformer.so", metavar='STRING',
                        help='the path of FasterTransformer pytorch t5 op library.')
    parser.add_argument('-model_path', '--model_path', type=str, default=None, metavar='STRING',
                        help='T5 model path.')
    parser.add_argument('-model', '--model', type=str, default="t5-small", metavar='STRING',
                        help='T5 model size. Only used when --model_path=None')
    # assume checkpoint config is also in the same path
    parser.add_argument('--ckpt_path', type=str, help='path to the checkpoint file.')
    parser.add_argument('-max_ite', '--max_iteration', type=int, default=100000, metavar='NUMBER',
                        help='Maximum iteraiton for translation, default is 100000 (as large as possible to run all test set).')
    parser.add_argument("--fifo_path", default = "ft_fifo", help = "Fifo path"\
    "used for IPC inside the FT process.")
    parser.add_argument("--verbose", action="store_true", help="Provide verbose messages")
    parser.add_argument("--port", type = int, default = 50051)
    parser.add_argument("--next_port", type = int, default = 50052)
    args = parser.parse_args()
    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=log_format)
    service = LmRetrieverServicer(vars(args))
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers = 1))
    lm_retriever_pb2_grpc.add_LmRetrieverServicer_to_server(service, server)
    port = args.port
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    server.wait_for_termination()