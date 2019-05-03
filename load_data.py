"""
Methods for preprocessing twitter dataset.
Author: Alex Reese
"""

import re
import io
import gluonnlp as nlp
import mxnet as mx
from mxnet.base import MXNetError
import random


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu(0)
        _ = mx.nd.array([0], ctx=ctx)
    except MXNetError:
        ctx = mx.cpu()
    return ctx


ctx = try_gpu()


def load_tsv_to_array(filepath):
    """
    Inputs: filepath to tsv file
    Outputs: list of lists with id, label, and tweet
    """
    array = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            split = line.split('\t')
            # Throw out "Can't Decide" entries as they're not useful
            if split[1] != "Can't Decide":
                array.append(split)
    return array


def load_dataset(train_file, val_file, test_file, max_length=64, embedding='glove.twitter.27B.50d'):
    """
    Inputs: training, validation and test files in TSV format
    Outputs: vocabulary (with attached embedding), training, validation and test datasets ready for neural net training
    """
    train_array = load_tsv_to_array(train_file)
    val_array = load_tsv_to_array(val_file)
    test_array = load_tsv_to_array(test_file)

    vocab = build_vocabulary(train_array, val_array, test_array, embedding='glove.twitter.27B.50d')
    train_dataset = preprocess_dataset(train_array, vocab, max_length)
    val_dataset = preprocess_dataset(val_array, vocab, max_length)
    test_dataset = preprocess_dataset(test_array, vocab, max_length)
    return vocab, train_dataset, val_dataset, test_dataset


def tokenize_text(array, all_tokens):
    """
    Inputs: list of lists with tweet data and list with all tokens across all datasets.
    Outputs: list with texts tokenized
    """
    tokenizer = nlp.data.SpacyTokenizer()
    for i, instance in enumerate(array):
        id_num, label, text = instance
        tokens = tokenizer(text.lower())
        all_tokens.extend(tokens)
        array[i] = [id_num, label, tokens]


def build_vocabulary(tr_array, val_array, tst_array, embedding='glove.twitter.27B.50d'):
    """
    Inputs: arrays representing the training, validation and test data
    Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """
    all_tokens = []
    tokenize_text(tr_array, all_tokens)
    tokenize_text(val_array, all_tokens)
    tokenize_text(tst_array, all_tokens)
    counter = nlp.data.count_tokens(all_tokens)
    vocab = nlp.Vocab(counter)
    glove = nlp.embedding.create('glove', source=embedding)
    vocab.set_embedding(glove)
    # If token not in vocab, add with randomized embedding
    for token in all_tokens:
        if not vocab.embedding[token].asnumpy().any():
            random_embed = mx.nd.array([random.uniform(-1.0, 1.0) for c in vocab.embedding['embed']], ctx=ctx)
            vocab.embedding[token] = random_embed
    return vocab


def _preprocess(x, vocab, max_len):
    """
    Inputs: data instance x (tokenized), vocabulary, maximum length of input (in tokens)
    Outputs: data mapped to token IDs, with corresponding label
    """
    id_num, label, text_tokens = x
    text_tokens = text_tokens[:max_len]
    data = vocab[text_tokens]
    return label, data


def preprocess_dataset(dataset, vocab, max_len):
    # Applies _preprocess to whole dataset
    preprocessed_dataset = [_preprocess(x, vocab, max_len) for x in dataset]
    return preprocessed_dataset


class BasicTransform(object):
    """
    This is a callable object used by the transform method for a dataset. It will be
    called during data loading/iteration.

    Parameters
    ----------
    labels : list string
        List of the valid strings for classification labels
    max_len : int, default 64
        Maximum sequence length - longer seqs will be truncated and shorter ones padded
    """
    def __init__(self, labels, max_len=64):
        self._max_seq_length = max_len
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i

    def __call__(self, label, data):
        label_id = self._label_map[label]
        padded_data = data + [0] * (self._max_seq_length - len(data))
        return mx.nd.array(padded_data, dtype='int32', ctx=ctx), mx.nd.array([label_id], ctx=ctx, dtype='int32')
