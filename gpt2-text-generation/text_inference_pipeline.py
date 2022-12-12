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

import argparse

executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "./exe_cache/")

def create_args(model_name_or_path, single_ipu, layers_per_ipu, matmul_proportion,prompt=None, **kwargs):
    args = argparse.Namespace(
        batch_size=1,
        device_iterations=1,
        fp16=True,
        input_len=150,
        output_len=256,
        prompt=prompt,
        repetition_penalty=2.0,
        save_samples_path=False,
        stop_token='#',
        temperature=1.2,
        tokenizer_type=0,
        topk=1,
        model_name_or_path=model_name_or_path,
        layers_per_ipu = layers_per_ipu,
        matmul_proportion = matmul_proportion,
        single_ipu=single_ipu,
        )
    
    for i in kwargs:
        if i in args:
            setattr(args,i,kwargs[i])
        else:
            raise KeyError(f"'{i}' is not a valid argument to the text inference pipeline")
    
    return args


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
            torch.int32), dynamic_mask[:, :-1]), dim=-1)

        return next_token, next_dynamic_mask

    def sharding_mapping(self):
        self.model.wte = poptorch.BeginBlock(self.model.wte, "emb", ipu_id=0)

        layer_ipu = _get_layer_ipu(self.args.layers_per_ipu)
        for index, layer in enumerate(self.model.h):
            ipu = layer_ipu[index]
            self.model.h[index] = poptorch.BeginBlock(
                layer, f"Encoder{index}", ipu_id=ipu)
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
    opts.enableExecutableCaching(executable_cache_dir)
    opts.randomSeed(42)
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


def get_input(tokenizer, args):
    #Get input and save it if nessercary
    if args.prompt is not None:
        text = args.prompt
    else:
        raise ValueError("You must provide a prompt to run inference, please set the variable `prompt` when initialising your arguments.")
    
    input_len = len(text)
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    
    return text_ids, input_len

def run_model(text_ids, model, tokenizer, input_len, args):
    
    all_ids = np.array([[text_ids[0]] for _ in range(args.batch_size)])
    position_ids = torch.zeros(args.batch_size, 1).to(torch.int32)

    dynamic_mask = torch.zeros(
        args.batch_size, args.input_len+args.output_len).to(torch.int32)
    dynamic_mask[:, 0] = 1
    model_time = []
    
    input_ids = torch.ones(args.batch_size, 1).to(
    torch.int32)*text_ids.pop(0)
    for _ in range(args.input_len + args.output_len):
        start_time = time.time()
        input_ids, dynamic_mask = model(
            input_ids.to(torch.int32), dynamic_mask.to(torch.int32), position_ids)
        end_time = time.time()
        model_time.append(end_time - start_time)
        position_ids += 1
        if len(text_ids) > 0:
            input_ids = torch.ones(args.batch_size, 1).to(
                torch.int32) * text_ids.pop(0)
        all_ids = np.concatenate(
            (all_ids, input_ids.view(args.batch_size, -1).numpy()), axis=1)

    for batch in all_ids.tolist():
        text = tokenizer.decode(
            batch, clean_up_tokenization_spaces=True)
        text = text[: text.find(args.stop_token, input_len , len(text))
                            if args.stop_token else None]

    return text


def gpt2(args):
    
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

    return model , tokenizer

def initialise_model(args):
    model , tokenizer = gpt2(args)
    text_ids, input_len = get_input(tokenizer, args)
    prompt = run_model(text_ids, model, tokenizer, input_len, args)
    return prompt , model, tokenizer

def sentiment_analysis(prompt):
    args = create_args("gpt-2",False,None,None,prompt)
    prompt , model, tokenizer = initialise_model(args)
    user_input = "### Message: " + input() + " Sentiment:"
    args.prompt = prompt + user_input
    text_ids, input_len = get_input(tokenizer, args)
    model_output = run_model(text_ids, model, tokenizer, input_len, args)
    output = model_output[len(prompt):]
    return prompt, output

class Pipeline:
    def __init__(self, model_name_or_path = "gpt2",
                         single_ipu=True,
                         layers_per_ipu=None,
                         matmul_proportion=None,
                         prompt = None,
                         **kwargs
                         ):
        self.args = create_args(model_name_or_path,
                         single_ipu,
                         layers_per_ipu,
                         matmul_proportion,
                         prompt,
                         **kwargs)
        self.output , self.model, self.tokenizer = initialise_model(self.args)
    
    def __call__(self, prompt:str) -> str:
        self.args.prompt = prompt
        text_ids, input_len = get_input(self.tokenizer, self.args)
        model_output = run_model(text_ids, self.model, self.tokenizer, input_len, self.args)
        model_output = model_output.replace(self.args.stop_token, "\n")
        return model_output