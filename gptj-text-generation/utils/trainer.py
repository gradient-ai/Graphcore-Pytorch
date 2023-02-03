# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import reduce

from config import GPTJConfig
from transformers.models.gptj import GPTJForCausalLM

from finetuning_mnli import finetuning_mnli
from run_finetuning_mnli import training
from run_mnli_validation import validate
from inference import inference

from modelling.hf_mapping import load_lm_to_hf
from typing import Optional
from torch.utils.data import Dataset


class MNLIFinetuningTrainer:
    def __init__(
        self,
        config: GPTJConfig,
        pretrained: GPTJForCausalLM,
        dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        eval_config: Optional[GPTJConfig] = None,
        tokenizer: Optional = None,
    ):
        inference_data = (eval_dataset, eval_config, tokenizer)
        if any(inference_data) and not all(inference_data):
            raise ValueError("If you want to dovalidation you need to specify a dataset, a config, and a tokenizer.")

        self.config = config
        self.train_session = finetuning_mnli(config)
        self.pretrained = pretrained
        self.dataset = dataset

        self.eval_config = eval_config
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.inference_session = self.__build_inference_session()

    def train(self):
        with self.train_session:
            training(self.config, self.train_session, self.pretrained, self.dataset)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_config: Optional[GPTJConfig] = None,
        tokenizer: Optional = None,
        checkpoint_dir: Optional[str] = None,
    ):
        inference_data = (eval_dataset, eval_config, tokenizer)
        if any(inference_data) and not all(inference_data):
            raise ValueError("If you want to do validation you need to specify a dataset, a config, and a tokenizer.")

        self.eval_config = eval_config
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.inference_session = self.__build_inference_session()

        if checkpoint_dir:
            self.inference_session.load_checkpoint(checkpoint_dir)
        else:
            self.inference_session.load_from_session(self.train_session)

        with self.inference_session:
            validate(self.eval_config, self.inference_session, self.eval_dataset, self.tokenizer)

    def save_hf_checkpoint(self, hf_ckpt_dir: str, ckpt_load_path: Optional[str] = None) -> GPTJForCausalLM:
        """
        Saves a checkpoint in Hugging Face format, which can then be loaded using Hugging Face API:
            ```
            AutoModelForCausalLM.from_pretrained(hf_ckpt_dir)
            ```
        Args:
            - hf_ckpt_dir (str): path to save the Hugging Face checkpoint
            - ckpt_load_path (str, Optional): path of a specific checkpoint. Default is None, meaning that the latest weights are saved.

        Returns:
            - GPTJForCausalLM: finetuned Hugging Face model
        """
        self.train_session.state = self.train_session.state.fwd
        if ckpt_load_path:
            self.train_session.load_checkpoint(ckpt_load_path)
        finetuned = load_lm_to_hf(self.train_session, self.pretrained)
        finetuned.save_pretrained(hf_ckpt_dir)
        return finetuned

    def __build_inference_session(self):
        self.inference_session = None
        if self.eval_config:
            if self.eval_dataset is None:
                raise ValueError("An evaluation dataset must be provided")
            if self.tokenizer is None:
                raise ValueError("A tokenizer must be provided")
            max_len = reduce(lambda l, e: max(l, len(e["input_ids"])), self.eval_dataset, 0)
            self.eval_config.model.sequence_length = max_len + self.eval_config.inference.output_length
            self.inference_session = inference(self.eval_config)
        return self.inference_session
