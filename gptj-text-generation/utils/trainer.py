# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Union
from functools import reduce

from config import GPTJConfig
from transformers.models.gptj import GPTJForCausalLM

from finetuning_mnli import finetuning_mnli
from run_finetuning_mnli import training
from run_mnli_validation import run_validation
from inference import inference

from modelling.hf_mapping import load_lm_to_hf
from typing import Optional, Callable
from torch.utils.data import Dataset
from popxl_addons.utils import timer

from .setup import xl_hf_config_check

# TODO proper type hints & doc


class MNLIFinetuningTrainer:
    def __init__(
        self,
        config: GPTJConfig,
        pretrained: Union[GPTJForCausalLM, str],
        dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        eval_config: Optional[GPTJConfig] = None,
        tokenizer: Optional = None,
        metric: Optional = None,
        process_answers_func: Optional[Callable] = None,
    ):
        self.config = config
        self.train_session = finetuning_mnli(config)
        # hf_model = "EleutherAI/gpt-j-6B"
        if isinstance(pretrained, str):
            with timer("Loading HuggingFace model"):
                pretrained = GPTJForCausalLM.from_pretrained(pretrained)
        xl_hf_config_check(config, pretrained.config)
        self.pretrained = pretrained
        self.dataset = dataset

        self.eval_config = eval_config
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.inference_session = self.__build_inference_session()
        self.metric = metric
        self.process_answers_func = process_answers_func

    def train(self):
        with self.train_session:
            training(self.config, self.train_session, self.pretrained, self.dataset)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_config: Optional[GPTJConfig] = None,
        tokenizer: Optional = None,
        metric: Optional = None,
        process_answers_func: Optional[Callable] = None,
    ):
        if tokenizer:
            self.tokenizer = tokenizer

        assert self.tokenizer is not None, "A tokenizer must be provided for evaluation"

        if eval_dataset:
            self.eval_dataset = eval_dataset
        assert self.eval_dataset is not None, "A dataset must be provided for evaluation"

        if metric:
            self.metric = metric
        assert self.metric is not None, "A metric must be provided for evaluation"

        if eval_config:
            self.eval_config = eval_config
            self.inference_session = self.__build_inference_session()

        if process_answers_func:
            self.process_answers_func = process_answers_func

        with self.inference_session:
            answers = run_validation(self.eval_config, self.inference_session, self.eval_dataset, self.tokenizer, self.train_session)
            formatted_answers = self.process_answers_func(answers)

        metrics = metric.compute(predictions=formatted_answers, references=dataset["label"])
        logging.info(metrics)
        return metrics

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
