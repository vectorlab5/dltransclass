"""Baseline implementations used in the paper.

The naming follows Section 4.1 of the manuscript:
    - FastText, TextCNN
    - BERT-base, BERT-Concat, RoBERTa-base, BERT-CNN-BiGRU
    - Hier-BERT, BERT-MTM-LSTM, HDGAT
    - LCSH-DeBERTa, MetA-MARC
    - Mamba-Doc, LongLoRA

Each baseline lives in its own submodule. Wrappers expose a common
``forward(batch) -> logits`` interface so scripts can swap baselines via the
config file.
"""

from dltransclass.baselines.bert_concat import BERTConcat

__all__ = ["BERTConcat"]
