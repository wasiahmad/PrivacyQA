# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/vector.py

import torch
from nqa.inputters import constants


def vectorize(ex, model):
    """Torchify a single example (an object of class Example).
    """

    max_sent_len = model.args.max_sent_len
    src_toks = [constants.BERT_CLS_WORD] + ex['question'].bert_token + [constants.BERT_SEP_WORD] + \
               ex['sentence'].bert_token + [constants.BERT_SEP_WORD]
    src_toks = src_toks[:max_sent_len]
    source_ids = model.tokenizer.convert_tokens_to_ids(src_toks)

    type_ids = [0] * (len(ex['question'].bert_token) + 2) + \
               [1] * (len(ex['sentence'].bert_token) + 1)
    type_ids = type_ids[:max_sent_len]

    return {
        'id': ex['id'],
        'sent_id': ex['sentence'].id,
        'sentence': ex['sentence'].text,
        'ques_id': ex['question'].id,
        'question': ex['question'].text,
        'source_ids': source_ids,
        'type_ids': type_ids,
        'label': ex['one_label'],
        'raw_label': ex['label']  # an OrderedDict
    }


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    max_source_len = max([len(ex['source_ids']) for ex in batch])

    source_ids = torch.LongTensor(batch_size, max_source_len).zero_()
    source_mask = torch.LongTensor(batch_size, max_source_len).zero_()
    source_type_ids = torch.LongTensor(batch_size, max_source_len).zero_()
    source_pos_ids = None
    label = torch.LongTensor(batch_size)

    for bidx in range(batch_size):
        example = batch[bidx]
        label[bidx] = example['label']
        ex_len = len(example['source_ids'])
        source_ids[bidx, :ex_len].copy_(torch.LongTensor(example['source_ids']))
        source_type_ids[bidx, :ex_len].copy_(torch.LongTensor(example['type_ids']))
        source_mask[bidx, :ex_len].copy_(torch.ones(ex_len))

    return {
        'ids': [ex['id'] for ex in batch],
        'batch_size': batch_size,
        'sent_id': [ex['sent_id'] for ex in batch],
        'sentence': [ex['sentence'] for ex in batch],
        'ques_id': [ex['ques_id'] for ex in batch],
        'question': [ex['question'] for ex in batch],
        'source_ids': source_ids,
        'source_pos_ids': source_pos_ids,
        'source_type_ids': source_type_ids,
        'source_mask': source_mask,
        'label': label,
        'raw_label': [ex['raw_label'] for ex in batch]
    }
