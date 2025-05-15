# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#改名为
import logging
import string
from collections import Counter
from typing import Callable
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import regex
from rouge import Rouge
import re

rouge = Rouge()

logger = logging.getLogger(__name__)
RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
# Normalization and score functions from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _normalize_text(text):
    result = text.lower()
    result = RE_PUNC.sub(' ', result)
    result = RE_ART.sub(' ', result)
    result = ' '.join(result.split())

    return result


def recall(passages, ground_truths):
    total = len(ground_truths)
    passages = [p.lower() for p in passages]
    ground_truths = [g.lower() for g in ground_truths]
    hits = 0
    for g in ground_truths:
        hit_flag = False
        for p in passages:
            if g in p:
                hit_flag = True
                break
        hits += float(hit_flag)
    return hits/total

def em(prediction, ground_truth, normalize_fn):
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))

def exact_match_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([em(prediction, gt, normalize_fn) for gt in ground_truths])


def f1(prediction, ground_truth, normalize_fn):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def rouge_wrapper(prediction, ground_truth):
    try:
        result = rouge.get_scores(prediction, ground_truth, avg=True)
        return result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]
    except:
        return 0.0, 0.0, 0.0


def f1_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([f1(prediction, gt, normalize_fn) for gt in ground_truths])



def rouge_score(prediction, ground_truths):
    ground_truths = [x for x in ground_truths if len(x) > 0]
    if (
        len(prediction) == 0 or len(ground_truths) == 0
    ):  # check if empty prediction or if there is no hypothesis with len > 0
        return 0.0, 0.0, 0.0
    scores = [rouge_wrapper(prediction, gt) for gt in ground_truths]
    rouge1 = max(s[0] for s in scores)
    rouge2 = max(s[1] for s in scores)
    rougel = max(s[2] for s in scores)
    return rouge1, rouge2, rougel




def ngram_precision(refs, hyp, n):
    """计算n-gram精确度"""
    hyp_ngrams = Counter(tuple(hyp[i:(i + n)]) for i in range(len(hyp) - n + 1))
    ref_ngrams = Counter()
    for ref in refs:
        ref_ngrams.update(tuple(ref[i:(i + n)]) for i in range(len(ref) - n + 1))
    hit = sum((hyp_ngrams & ref_ngrams).values())
    total = sum(hyp_ngrams.values())
    return hit / total if total > 0 else 0

def brevity_penalty(refs, hyp):
    """计算短句惩罚"""
    ref_length = max(len(ref) for ref in refs)
    hyp_length = len(hyp)
    if hyp_length > ref_length:
        return 1
    return np.exp(1 - ref_length / hyp_length) if hyp_length > 0 else 0

def bleu_score(prediction, ground_truths, max_n=4):
    """计算BLEU分数"""
    hyp = tokenize(prediction)
    refs = [tokenize(ref) for ref in ground_truths]
    precisions = [ngram_precision(refs, hyp, n) for n in range(1, max_n + 1)]
    if min(precisions) == 0:
        return 0
    score = np.exp(sum(np.log(p) for p in precisions) / len(precisions))
    return score #* brevity_penalty(refs, hyp)
def tokenize(text):
    """将文本字符串分词为单词列表，使用空格作为分隔符"""
    return text.split()

def _bleu(prediction, ground_truths, n=4):
    # if (ref_tokens == ['cannotanswer']):
    #     if (hyp_tokens == ['cannotanswer']):
    #         return 1
    #     else:
    #         return 0
    # else:
    hyp_tokens = tokenize(normalize_answer(prediction))
    ref_tokens = [tokenize(normalize_answer(ref)) for ref in ground_truths]



    weights = [1 / n] * n
    score = sentence_bleu(ref_tokens, hyp_tokens, weights)

    return score