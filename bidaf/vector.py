# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/vector.py

import torch


def map_to_ids(tokens, word_dict, _type='word'):
    if _type == 'word':
        return [word_dict[w] for w in tokens]
    elif _type == 'char':
        return [word_dict.word_to_char_ids(w).tolist() for w in tokens]
    else:
        assert False


def vectorize(ex, model):
    """Torchify a single example {'id': , 'sentence': , 'question': , 'label': }.
    """

    max_sent_len = model.args.max_sent_len

    # Index words
    sentence_word_rep = torch.LongTensor(map_to_ids(ex['sentence'].word[:max_sent_len],
                                                    word_dict=model.src_dict))
    question_word_rep = torch.LongTensor(map_to_ids(ex['question'].word[:max_sent_len],
                                                    word_dict=model.src_dict))
    sentence_char_rep = None
    question_char_rep = None
    # Index chars
    if model.args.use_chars:
        sentence_char_rep = torch.LongTensor(map_to_ids(ex['sentence'].word[:max_sent_len],
                                                        word_dict=model.src_dict,
                                                        _type='char'))
        question_char_rep = torch.LongTensor(map_to_ids(ex['question'].word[:max_sent_len],
                                                        word_dict=model.src_dict,
                                                        _type='char'))

    return {
        'id': ex['id'],
        'sent_id': ex['sentence'].id,
        'sentence': ex['sentence'].text,
        'sentence_word_rep': sentence_word_rep,
        'sentence_char_rep': sentence_char_rep,
        'ques_id': ex['question'].id,
        'question': ex['question'].text,
        'question_word_rep': question_word_rep,
        'question_char_rep': question_char_rep,
        'label': ex['one_label'],
        'use_chars': model.args.use_chars,
        'raw_label': ex['label']  # an OrderedDict
    }


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    use_char = batch[0]['use_chars']
    if use_char:
        max_word_len = batch[0]['sentence_char_rep'].size(1)

    # --------- Prepare source tensors ---------

    sentence_word = [ex['sentence_word_rep'] for ex in batch]
    sentence_char = [ex['sentence_char_rep'] for ex in batch]
    question_word = [ex['question_word_rep'] for ex in batch]
    question_char = [ex['question_char_rep'] for ex in batch]

    # Batchify sources
    max_sentence_length = max([d.size(0) for d in sentence_word])
    sentence_len = torch.LongTensor(batch_size).zero_()
    sentence_word_rep = torch.LongTensor(batch_size,
                                         max_sentence_length).zero_()
    sentence_char_rep = torch.LongTensor(batch_size,
                                         max_sentence_length,
                                         max_word_len).zero_() if use_char else None

    max_question_length = max([d.size(0) for d in question_word])
    question_len = torch.LongTensor(batch_size).zero_()
    question_word_rep = torch.LongTensor(batch_size,
                                         max_question_length).zero_()
    question_char_rep = torch.LongTensor(batch_size,
                                         max_question_length,
                                         max_word_len).zero_() if use_char else None
    label = torch.LongTensor(batch_size).zero_()

    for i in range(batch_size):
        sentence_len[i] = sentence_word[i].size(0)
        question_len[i] = question_word[i].size(0)
        sentence_word_rep[i, :sentence_word[i].size(0)].copy_(sentence_word[i])
        question_word_rep[i, :question_word[i].size(0)].copy_(question_word[i])
        label[i] = batch[i]['label']
        if use_char:
            sentence_char_rep[i, :sentence_char[i].size(0), :].copy_(sentence_char[i])
            question_char_rep[i, :question_char[i].size(0), :].copy_(question_char[i])

    return {
        'ids': [ex['id'] for ex in batch],
        'batch_size': batch_size,
        'sent_id': [ex['sent_id'] for ex in batch],
        'sentence': [ex['sentence'] for ex in batch],
        'sentence_word_rep': sentence_word_rep,
        'sentence_char_rep': sentence_char_rep,
        'sentence_len': sentence_len,
        'ques_id': [ex['ques_id'] for ex in batch],
        'question': [ex['question'] for ex in batch],
        'question_word_rep': question_word_rep,
        'question_char_rep': question_char_rep,
        'question_len': question_len,
        'label': label,
        'raw_label': [ex['raw_label'] for ex in batch]
    }
