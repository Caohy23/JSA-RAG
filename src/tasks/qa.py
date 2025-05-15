# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

from metrics import exact_match_score, f1_score, normalize_answer, bleu_score,_bleu,rouge_score
from src.options import Options
from src.tasks.base import BaseTask


class Task(BaseTask):
    metrics = ["exact_match", "f1", "eval_loss","BLEU-4","BLEU-1","Rouge-1","Rouge-2","Rouge-L"]

    def __init__(self, opt: Options, *args, **kwargs):
        super().__init__()
        self.qa_prompt_format_str = opt.qa_prompt_format
        self.decoder_only = opt.decoder_only

    def get_qa_prompt(self, question: str) -> str:
        return self.qa_prompt_format_str.format(question=question)

    def process(self, example, *args, **kwargs):

        if "target" in example:
            target = example["target"]
        elif "answers" in example:
            target = random.choice(example["answers"])
        else:
            target = None

        if not "passages" in example:
            example["passages"] = [{"title": "", "text": ""}]

        example["metadata"] = example.get("metadata", {})
        example["query"] = self.get_qa_prompt(example["question"])
        if target is not None:
            example["target"] = f"<extra_id_0> {target}" if not self.decoder_only else f"{target}"

        return example

    def evaluation(self, prediction, ground_truths):
        Rouge_1,Rouge_2,Rouge_L = rouge_score(prediction, ground_truths)
        sample_metrics = {
            "exact_match": exact_match_score(prediction, ground_truths, normalize_answer),
            "f1": f1_score(prediction, ground_truths, normalize_answer),
            "BLEU-4":_bleu(prediction, ground_truths, 4),
            "BLEU-1":_bleu(prediction, ground_truths, 1),
            "Rouge-1":Rouge_1,
            "Rouge-2":Rouge_2,
            "Rouge-L":Rouge_L
        }
        return sample_metrics


