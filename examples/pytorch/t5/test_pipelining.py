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
from transformers import T5ForConditionalGeneration, T5Tokenizer # transformers-4.10.0-py3
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")

from examples.pytorch.t5.utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder
from examples.pytorch.t5.utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from examples.pytorch.decoding.utils.recover_bpe import recover_bpe

LOGGER = logging.getLogger(__name__)

gemm_data_type_mapping = {"fp32":0, "fp16":1, "bf16":2}
class TranslationResult(object):
    def __init__(self, name, frame_work):
        self.name = name
        self.frame_work = frame_work # FT or HF
        self.file_name = name + ".txt"

        self.token_list = []
        self.batch_ids_list = []
        self.batch_seq_len_list = []
        self.batch_num = 0
        self.execution_time = 0.0  # seconds
        self.sentence_num = 0
        self.token_num = 0
        self.bleu_score = None
    def __add__(self, other):
        self.token_num += other.token_num
        self.execution_time += other.execution_time
        self.batch_num += other.batch_num
        self.token_list.extend(other.token_list)
        return self

def run_translation(args_dict):
    torch.set_printoptions(precision=6)
    batch_size = args_dict['batch_size']
    beam_size = 1
    max_seq_len = args_dict['max_seq_len']
    source_file = args_dict["source"]
    tgt_file = args_dict["target"]
    beam_search_diversity_rate = 0.0
    topk = 1
    topp = 0.0
    tensor_para_size = 1
    pipeline_para_size = 1
    max_ite = args_dict['max_iteration']
    repetition_penalty = 1.0
    temperature = 1.0
    len_penalty = 0.0
    ## huggingface without bias and use relative position embedding
    ## relative position embedding -> 0, absolute position embedding -> 1
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

    result = TranslationResult("ft-samping", "FT")
    pipelining_result = TranslationResult("ft-sampling", "FT")
    
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
    # TODO: try other prompts and see what the LM produces
    with open(source_file, 'r') as f:
        src_text = recover_bpe(f.readlines())
        src_text = ["translate English to German: " + line.strip() for line in src_text]
    
    with open(tgt_file, 'r') as f:
        tgt_text = recover_bpe(f.readlines())

    # now repeat, but for the pipelined version 
    fifo_path = "ft_fifo"
    if os.path.exists(fifo_path):
        os.unlink(fifo_path)
    # make fifo
    os.mkfifo(fifo_path, 0x600)
    prev = 0
    # for now, just do 1 inference
    batch_id = -1
    queue = mp.Queue()
    full_translation_result = TranslationResult("pipelining", "FT")
    translation_results = []
    start_time = datetime.now()
    while prev < len(src_text):
        batch_id += 1
        p = mp.Process(target = read_from_pipe_and_tokenize,
                   args = (queue, 
                           fast_tokenizer, 
                           fifo_path,
                           max_seq_len, 
                           min(batch_size, len(src_text) - prev), 
                           batch_id))
        p.start()
        input_texts = src_text[prev:prev + batch_size]
        prev += batch_size
        input_token = tokenizer(input_texts, return_tensors = 'pt', padding =True)
        bad_words_list = None
        stop_words_list = None
        tmp_beam_size = beam_size
        ft_decoding_outputs, ft_decoding_seq_lens = ft_t5(input_token,
                                                                  None,
                                                                  tmp_beam_size,
                                                                  max_seq_len,
                                                                  topk,
                                                                  topp,
                                                                  beam_search_diversity_rate=beam_search_diversity_rate,
                                                                  is_return_output_log_probs=False,
                                                                  is_return_cum_log_probs=False,
                                                                  repetition_penalty=repetition_penalty,
                                                                  temperature=temperature,
                                                                  len_penalty=len_penalty,
                                                                  bad_words_list=bad_words_list,
                                                                  stop_words_list=stop_words_list,
                                                                  intermediate_fifo_file=fifo_path)
        p.join()
        translation_results.append(queue.get())
    stop_time = datetime.now()
    for x in translation_results:
        full_translation_result += x
    full_translation_result.execution_time = (stop_time -
                                              start_time).total_seconds()
    LOGGER.info(f"Finished full inference in"\
                f" {full_translation_result.execution_time}s.")

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

def read_from_pipe_and_tokenize(queue, 
                                fast_tokenizer, 
                                fifo_path, 
                                max_seq_len,
                                batch_size, 
                                batch_id):
    translation_result = TranslationResult(f"batch_{batch_id}", "HT")
    for idx in range(batch_size):
        translation_result.token_list.append([])
    # TODO: also write to new token stream to link to next output
    r = os.open(fifo_path, os.O_RDONLY)
    tokens_read = 0
    finished = [0 for _ in range(batch_size)]
    while sum(finished) < batch_size:
        index = read_int32_from_file(r)
        value = read_int32_from_file(r)
        index_done = read_bool_from_file(r)
        tokenized = fast_tokenizer.decode([value],
                                          skip_special_tokens = True)
        translation_result.token_list[index].append(tokenized)
        # TODO: put on a queue onto the next microservice in the pipeline
        finished[index] = index_done
    queue.put(translation_result) 

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
    parser.add_argument("--verbose", action="store_true", help="Provide verbose messages")
    args = parser.parse_args()
    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=log_format)
    run_translation(vars(args))

