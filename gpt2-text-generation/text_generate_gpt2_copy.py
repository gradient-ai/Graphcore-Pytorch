# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import ctypes
import logging
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import poptorch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

from ipu_options import load_custom_ops
from tools import _get_layer_ipu, str_to_bool
from model.optimized_gpt2_attn import OptimizedGPT2AttentionBuffer, OptimizedGPT2AttentionCache

MODEL_CONFIG = {'gpt2': 'config/config.json', 'gpt2-medium': 'config/config_medium.json',
                'gpt2-large': 'config/config_large.json', 'gpt2-xl': 'config/config_xl.json'}

logging.basicConfig(level=logging.INFO, format="%(message)s")


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name-or-path", type=str, default="gpt2",
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CONFIG.keys()))
    parser.add_argument('--tokenizer-type', type=int, default=0,
                        help='0: transformers.tokenizer, 1: Megatron.tokenizer')
    parser.add_argument('--temperature', default=1.2,
                        type=float, required=False, help='temperature')
    parser.add_argument('--repetition-penalty', default=2.0,
                        type=float, required=False, help="repetition_penalty")
    parser.add_argument('--topk', default=4, type=int,
                        required=False, help='topk to choice')
    parser.add_argument('--save-samples-path', type=str, default=None,
                        required=False, help="path to save generated text")
    parser.add_argument('--prompt', type=str, default=None,
                        help='Prompt as input')
    parser.add_argument('--input-len', type=int, default=64,
                        help='Maximum input length')
    parser.add_argument('--output-len', type=int, default=128,
                        help='Maximum length of generated text')
    parser.add_argument("--batch-size", type=int, default=1,
                        help='batch size (default = 1)')
    parser.add_argument('--device-iterations', type=int,
                        default=1, help='device iterations (default = 1)')
    parser.add_argument("--single-ipu", type=str_to_bool, nargs="?",
                        const=True, default=False, help="single ipu or not")
    parser.add_argument('--layers-per-ipu', type=int, default=3,
                        nargs="+", help='Number of decoder layers per pipeline stage.')
    parser.add_argument("--matmul-proportion", type=float, nargs="+",
                        help="Relative IPU memory proportion size allocated for matmul")
    parser.add_argument("--fp16", type=str_to_bool, nargs="?",
                        const=True, default=False, help="run model in fp16")
    parser.add_argument("--stop-token", type=str, default="<|endoftext|>",
                        help='Token at which text generation is stopped')
    return parser.parse_args()


class GPT2Wrapper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.count = args.output_len
        self.args = args
        if args.model_name_or_path:
            self.model = GPT2Model.from_pretrained(args.model_name_or_path)
        else:
            raise RuntimeError("--model-name-or-path must be set.")
        self.nop = poptorch.nop
        self.optimize()
        if not args.single_ipu:
            self.sharding_mapping()

    def optimize(self):
        self.model.config.batch = self.args.batch_size
        self.model.config.seq = self.args.input_len + self.args.output_len
        self.model.config.input_len = self.args.input_len
        self.model.config.output_len = self.args.output_len
        self.model.config.activation_function = "gelu"
        inner_dim = self.model.config.n_inner if self.model.config.n_inner is not None else 4 * \
            self.model.config.hidden_size
        for layer in self.model.h:
            GPT2Attn = OptimizedGPT2AttentionBuffer(self.model.config)
            MLP = GPT2MLP(inner_dim, self.model.config)
            GPT2Attn.load_state_dict(layer.attn.state_dict(), strict=False)
            MLP.load_state_dict(layer.mlp.state_dict(), strict=False)
            layer.attn = GPT2Attn
            layer.mlp = MLP

    def forward(self, context, dynamic_mask, position_ids):
        hidden_states = self.model(context, attention_mask=dynamic_mask,
                                    position_ids=position_ids, past_key_values=None, return_dict=False)
        hidden_states_ = self.nop(hidden_states[0])
        next_token_logits = torch.matmul(
            hidden_states_, self.model.wte.weight.T).view(self.args.batch_size, -1)
        (next_token_value, next_token) = torch.topk(
            next_token_logits, self.args.topk)
        # We simply do a random selection after topk to avoid repetitions
        # Notice: Here we use 'argmax' + 'randn' instead of 'randint' which is unsupported.
        random_choice_idx = torch.argmax(torch.randn((1, self.args.topk)), axis=1)
        next_token = next_token[:, random_choice_idx]

        next_dynamic_mask = torch.cat((torch.ones(self.args.batch_size, 1).to(
            torch.int64), dynamic_mask[:, :-1]), dim=-1)

        return next_token, next_dynamic_mask

    def sharding_mapping(self):
        print("-------------------- Device Allocation --------------------")
        print("Embedding  --> IPU 0")
        self.model.wte = poptorch.BeginBlock(self.model.wte, "emb", ipu_id=0)

        layer_ipu = _get_layer_ipu(self.args.layers_per_ipu)
        for index, layer in enumerate(self.model.h):
            ipu = layer_ipu[index]
            self.model.h[index] = poptorch.BeginBlock(
                layer, f"Encoder{index}", ipu_id=ipu)
            print(f"Layer {index:<2} --> IPU {ipu}")
        self.nop = poptorch.BeginBlock(self.nop, ipu_id=0)

def get_options(args):
    if args.single_ipu:
        mem_prop = {'IPU0': 0.2}
    else:
        mem_prop = {
            f'IPU{i}': args.matmul_proportion[i]
            for i in range(len(args.matmul_proportion))
        }

    opts = poptorch.Options().deviceIterations(args.device_iterations)
    opts.autoRoundNumIPUs(True)
    opts.setAvailableMemoryProportion(mem_prop)
    opts._Popart.set("saveInitializersToFile", "weights.bin")
    opts.enableExecutableCaching("./exe")
    if not args.single_ipu:
        opts.setExecutionStrategy(poptorch.ShardedExecution())
    return opts

def get_tokenizer(args):
    if args.tokenizer_type == 0:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    else:
        from tokenizer import build_megatron_tokenizer
        tokenizer = build_megatron_tokenizer(
            vocab_file="./tokenizer/gpt2-vocab-50256.json", merge_file="./tokenizer/gpt2-merges-50256.txt")
    return tokenizer

#def define_model():

def get_input(tokenizer):
    #Get input and save it if nessercary
    if args.prompt is not None:
        text = args.prompt
    else:
        text = input("Input: ")
        input_len = len(text)
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        txt_len = len(text_ids)
        if args.input_len < txt_len:
            print("Input text length {0} larger than limit {1}".format(
                txt_len, args.input_len))

        if args.save_samples_path:
            samples_file.write("Input: {}\n".format(text))
    return text_ids, txt_len, input_len

def run_model(text_ids, txt_len, model, tokenizer, input_len):
    input_ids_all = torch.tensor(text_ids).long()
    all_ids = np.array([[text_ids[0]] for _ in range(args.batch_size)])
    position_ids = torch.zeros(args.batch_size, 1).to(torch.int64)

    dynamic_mask = torch.zeros(
        args.batch_size, args.input_len+args.output_len).to(torch.int64)
    dynamic_mask[:, 0] = 1
    model_time = []

    input_ids = torch.ones(args.batch_size, 1).to(
    torch.int64)*text_ids.pop(0)
    for _ in range(args.input_len + args.output_len):
        start_time = time.time()
        input_ids, dynamic_mask = model(
            input_ids.to(torch.int64), dynamic_mask.to(torch.int64), position_ids)
        end_time = time.time()
        model_time.append(end_time - start_time)
        position_ids += 1
        if len(text_ids) > 0:
            input_ids = torch.ones(args.batch_size, 1).to(
                torch.int64) * text_ids.pop(0)
        all_ids = np.concatenate(
            (all_ids, input_ids.view(args.batch_size, -1).numpy()), axis=1)
    logging.info(
        "latency avg per sentence: {0} ms/sentence_({1})".format(np.sum(model_time[1:])*1000, args.output_len))
    logging.info(
        "Per-token latency avg: {} ms/token".format(np.mean(model_time[1:])*1000))
    logging.info("Batch size: {0}; Input length {1}; Output length {2}, throughput: {3} samples/sec \n".format(
        args.batch_size, txt_len, args.output_len, args.batch_size / np.sum(model_time[1:])))
    for batch in all_ids.tolist():
        text = tokenizer.decode(
            batch, clean_up_tokenization_spaces=True)
        text = text[: text.find(args.stop_token, input_len , len(text))
                            if args.stop_token else None]
        logging.info(text)
        if args.save_samples_path:
            samples_file.write("Output: {}\n".format("".join(text)))


def main_gpt2(args):
    print(args)
    # custom op
    load_custom_ops()

    # Get poptorch options
    opts = get_options(args)

    #Get tokeniser
    tokenizer = get_tokenizer(args)

    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path +
                            '/samples.txt', 'a', encoding='utf8')
        samples_file.write("Text generator record{}:\n".format(datetime.now()))

    #Define model
    model = GPT2Wrapper(args)
    if args.fp16:
        model.half()
    model = poptorch.inferenceModel(model.eval(), opts)
    text_ids, txt_len, input_len = get_input(tokenizer)



    if args.save_samples_path:
        samples_file.close()

    run_model(text_ids, txt_len, model, tokenizer, input_len)


if __name__ == '__main__':
    args = set_args()
    main_gpt2(args)
