# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
""" A module which provides an interface to this application
resembling the Hugging Face Transformers Pipeline.
"""
from typing import Union, List, Optional, Any

import abc

import logging
from functools import partial
import re

import torch
import datasets

import popxl
from popxl.utils import to_numpy
from tqdm import tqdm
from inference import inference
from modelling.embedding import GPTJEmbeddingsTP
from modelling.hf_mapping import hf_mapping_lm_tp
from popxl_addons import timer
from utils.utils import tensor_parallel_input, repeat
from utils.inference import batch_inference
from config import GPTJConfig

from transformers.models.gptj import GPTJForCausalLM
from transformers import AutoTokenizer
from data import mnli_data

import popart
import popxl


class BasicPipeline:
    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, prompt: Union[str, List[str]]) -> Any:
        pass


def unwrap(dl):
    for example in dl:
        yield torch.tensor(example["input_ids"], dtype=torch.long)


def encode_for_inference(dataset: datasets.Dataset, tokenizer: AutoTokenizer):
    tokenized_examples = []
    for example in dataset["text"]:
        tokenized_example = tokenizer.encode(example, return_tensors="pt").squeeze()
        tokenized_examples.append(tokenized_example)
    return {
        "input_ids": tokenized_examples,
        "label": [None for _ in range(len(tokenized_examples))],
    }


class IPUGPTJPipeline(BasicPipeline):
    def __init__(
        self,
        config: GPTJConfig,
        hf_gptj_checkpoint: str,
        *args,
        sequence_length=None,
        output_length=128,
        print_live=False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if sequence_length is not None:
            config.model.sequence_length = sequence_length

        logging.info(f"Creating session")
        session: popxl.Session = inference(config)
        logging.info(f"Downloading '{hf_gptj_checkpoint}' pretrained weights and tokenizer")
        hf_model = GPTJForCausalLM.from_pretrained(hf_gptj_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(hf_gptj_checkpoint)
        tokenizer.add_special_tokens({"pad_token": "<|extratoken_1|>"})

        if config.model.dtype == popxl.float16:
            hf_model.half()
        with timer("Loading HF pretrained model to IPU"):
            weights = hf_mapping_lm_tp(config, session, hf_model)
            session.write_variables_data(weights)
        self.tokenizer = tokenizer
        self.model = hf_model
        self.config = config
        self.session = session
        self.output_length = output_length
        self.print_live: bool = print_live
        self._termination_token: Optional[torch.Tensor] = None

    def next_token(self, inputs, lengths):
        tp = self.config.execution.tensor_parallel
        rf = self.config.execution.tensor_parallel * self.config.execution.data_parallel
        data_map = {}
        words = to_numpy(inputs, self.session.inputs.words.dtype).reshape(-1, *self.session.inputs.words.shape)
        data_map[self.session.inputs.words] = tensor_parallel_input(
            words, tp, rf, partial(GPTJEmbeddingsTP.offset_input, config=self.config)
        ).squeeze()
        data_map[self.session.inputs.last_token_indices] = repeat(lengths - 1, tp, axis=0)
        # identical for all tp, take first
        next_token_id = self.session.run(data_map)[self.session.outputs.next_token][0]
        # find if the termination string was encountered and replace it with the end of string
        # token of the tokenizer
        if self._termination_token is not None:
            shape = self._termination_token.shape[0]
            for i, (input_, length) in enumerate(zip(inputs, lengths)):
                if shape > length or shape > len(input_):
                    continue
                if (input_[length - shape + 1 : length] == self._termination_token[:-1]).all() and next_token_id[
                    i
                ] == self._termination_token[-1]:
                    next_token_id[i] = self.tokenizer.eos_token_id
        if self.print_live:
            print(self.tokenizer.decode(next_token_id[0]), end="")
        return torch.LongTensor(next_token_id)

    def __call__(
        self,
        prompt: Union[str, List[str], datasets.Dataset],
        *args,
        output_length: Optional[int] = None,
        print_live: Optional[bool] = None,
        terminate_on_string: Optional[str] = None,
    ):
        super().__call__(prompt)
        if print_live is not None:
            self.print_live = print_live
        if isinstance(prompt, str):
            prompt = [prompt]
        # Preprocess the data including batching it
        output_length = self.output_length if output_length is None else output_length
        micro_batch = self.config.execution.micro_batch_size

        if isinstance(prompt, datasets.Dataset):
            data = prompt
        else:
            data = datasets.Dataset.from_dict({"text": prompt})
        data = data.map(
            encode_for_inference,
            batched=True,
            remove_columns=data.column_names,
            load_from_cache_file=False,
            fn_kwargs={"tokenizer": self.tokenizer},
        )

        logging.info("Attach to IPUs")
        self.session.__enter__()
        logging.info("Start inference")
        if self.print_live:
            print(f"Prompt: '{prompt[0] if isinstance(prompt, list) else prompt[0]['text']}'", end="")
        if terminate_on_string:
            self._termination_token = encode_for_inference({"text": [terminate_on_string]}, self.tokenizer)[
                "input_ids"
            ][0].squeeze()
        answers = batch_inference(
            unwrap(data),
            self.next_token,
            self.config.model.sequence_length,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            output_length=output_length,
            micro_batch_size=micro_batch,
        )
        self._termination_token = None
        text_outputs = []
        for a in answers:
            text = self.tokenizer.decode(a)
            text_outputs.append(text)
        return text_outputs

    def detach(self):
        was_attached_or_device = self.session._was_attached_stack.pop()

        # self.session.weights_to_host()
        self.session._device.detach()
        self.session._pb_session.setEngineIsLoaded(False)

        # If a DeviceInfo was stored in the stack then restore it.
        if isinstance(was_attached_or_device, popart.DeviceInfo):
            self.session._set_device(was_attached_or_device)

    @classmethod
    def from_gptj_pipeline(cls, other: "IPUGPTJPipeline"):
        """Create a new pipeline with the same model, config, tokenizer and session
        This can be used to quickly reuse the GPTJ session and IPU for a different task"""
        new_pipeline = cls.__new__(cls)
        new_pipeline.tokenizer = other.tokenizer
        new_pipeline.model = other.model
        new_pipeline.config = other.config
        new_pipeline.session = other.session
        new_pipeline.output_length = other.output_length
        new_pipeline.print_live = other.print_live
        new_pipeline._termination_token = other._termination_token
        return new_pipeline


class GPTJEntailmentPipeline(IPUGPTJPipeline):
    """A generative pipeline for"""

    def __call__(
        self,
        premise: Union[str, List[str]],
        hypothesis: Union[str, List[str]],
        *args,
        output_length: Optional[int] = 10,
        print_live: bool = True,
        **kwargs,
    ):
        data = datasets.Dataset.from_dict(
            {
                "premise": [premise] if isinstance(premise, str) else premise,
                "hypothesis": [hypothesis] if isinstance(hypothesis, str) else hypothesis,
            }
        )
        data = data.map(
            mnli_data.form_validation_prompts,
            remove_columns=data.column_names,
        )
        print(len(data))
        self.raw_out = super().__call__(data, *args, output_length=output_length, print_live=print_live, **kwargs)
        find_eos = re.compile(re.escape(self.tokenizer.eos_token))
        processed = []
        for generated in self.raw_out:
            first_line = generated.splitlines()[0].strip()
            has_eos = find_eos.search(first_line)
            if has_eos:
                processed.append(first_line[: has_eos.start()])
            else:
                processed.append(first_line)

        return processed
