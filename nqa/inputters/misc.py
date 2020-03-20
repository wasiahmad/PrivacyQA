# ref: https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/extract_features.py


def convert_example_to_features(example, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    assert isinstance(example, str)
    tokens_a = tokenizer.tokenize(example)

    tokens = list()
    tokens.append("[CLS]")
    for token in tokens_a:
        tokens.append(token)
    tokens.append("[SEP]")

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    return {
        'tokens': tokens,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'input_len': len(tokens)
    }
