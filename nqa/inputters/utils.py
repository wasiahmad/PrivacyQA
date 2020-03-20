import json
import os
import logging
from tqdm import tqdm
from collections import Counter, OrderedDict

from nqa.utils.misc import count_file_lines
from nqa.inputters import constants
from nqa.objects import Sentence
from nqa.inputters.vocabulary import Vocabulary, UnicodeCharsVocabulary

logger = logging.getLogger(__name__)
MAX_SEQUENCE_LENGTH = 512


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def get_company_names(src_dir):
    filenames = os.listdir(src_dir)
    com_names = set()
    for f in filenames:
        com_names.add(os.path.splitext(f)[0])

    return list(com_names)


def read_conll_file(filename, uncase):
    sentences = list()
    with open(filename) as f:
        word, lemma, pos, head, deprel, spaceafter = [], [], [], [], [], []
        sent_id = None
        for line in f:
            line = line.strip()
            if line:
                tokens = line.split('\t')
                if sent_id is None:
                    sent_id = int(tokens[0])
                word.append(tokens[2].lower() if uncase else tokens[2])
                lemma.append(tokens[3])
                pos.append(tokens[4])
                if tokens[6] == '_':
                    head.append(-1)
                    deprel.append('_')
                else:
                    toks = tokens[6].split(':')
                    assert len(toks) == 2
                    head.append(int(toks[0]))
                    deprel.append(toks[1])
                isspace = True if tokens[7] == '_' else False
                spaceafter.append(isspace)
            else:
                # an empty line
                assert sent_id is not None
                sent = Sentence(sent_id)
                sent.word = word
                sent.lemma = lemma
                sent.pos = pos
                sent.head = head
                sent.deprel = deprel
                sent.spaceafter = spaceafter
                sentences.append(sent)
                sent_id = None
                word, lemma, pos, head, deprel, spaceafter = [], [], [], [], [], []

    if sent_id is not None:
        sent = Sentence(sent_id)
        sent.word = word
        sent.lemma = lemma
        sent.pos = pos
        sent.head = head
        sent.deprel = deprel
        sent.spaceafter = spaceafter
        sentences.append(sent)

    return sentences


def read_labels(filename):
    labels = list()
    with open(filename) as f:
        for line in f:
            tokens = line.strip().split('\t')
            assert len(tokens) >= 3
            one_label = OrderedDict()
            for anno_id, anno in enumerate(tokens[2:]):
                if anno == '_':
                    one_label[anno_id] = []
                else:
                    one_label[anno_id] = [int(v) for v in anno.split(',')]
            assert len(one_label) >= 1
            labels.append(one_label)

    return labels


def load_data(src_dir,
              uncase=True,
              dataset_name='privacyQA',
              max_examples=-1):
    company_names = get_company_names(src_dir)

    examples = []
    for cname in company_names:
        sentences = read_conll_file(os.path.join(src_dir + '%s.sentence' % cname), uncase)
        questions = read_conll_file(os.path.join(src_dir + '%s.question' % cname), uncase)
        labels = read_labels(os.path.join(src_dir + '%s.label' % cname))
        assert len(questions) == len(labels)

        if dataset_name in ['privacyQA']:
            for ques, one_label in zip(questions, labels):
                for sent in sentences:
                    ex = dict()
                    ex['id'] = cname
                    ex['sentence'] = sent
                    ex['question'] = ques
                    ex['one_label'] = 1 if sent.id in one_label[0] else 0
                    ex['label'] = one_label
                    examples.append(ex)
        else:
            raise NotImplementedError

    # this is mainly for debugging purpose
    if max_examples != -1:
        examples = examples[:max_examples]

    return examples


# ------------------------------------------------------------------------------
# Dictionary building
# ------------------------------------------------------------------------------


def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file) as f:
        for line in tqdm(f, total=count_file_lines(embedding_file)):
            w = Vocabulary.normalize(line.rstrip().split(' ')[0])
            words.add(w)

    words.update([constants.PAD_WORD, constants.UNK_WORD])
    return words


def load_words(args, examples, fields, dict_size=None):
    """Iterate and index all the words in examples (documents + questions)."""

    def _insert(iterable):
        words = []
        for w in iterable:
            w = Vocabulary.normalize(w)
            if valid_words and w not in valid_words:
                continue
            words.append(w)
        word_count.update(words)

    if args.restrict_vocab and args.embedding_file:
        logger.info('Restricting to words in %s' % args.embedding_file)
        valid_words = index_embedding_words(args.embedding_file)
        logger.info('Num words in set = %d' % len(valid_words))
    else:
        valid_words = None

    word_count = Counter()
    for ex in tqdm(examples):
        for field in fields:
            _insert(ex[field].word)

    # -2 to reserve spots for PAD and UNK token
    dict_size = dict_size - 2 if dict_size and dict_size > 2 else dict_size
    most_common = word_count.most_common(dict_size)
    words = set(word for word, _ in most_common)
    return words


def build_word_dict(args, examples, fields, dict_size=None):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Vocabulary()
    words = load_words(args, examples, fields, dict_size)
    for w in words:
        word_dict.add(w)
    return word_dict


def build_word_and_char_dict(args, examples, fields, dict_size=None):
    """Return a dictionary from question and document words in
    provided examples.
    """
    words = load_words(args, examples, fields, dict_size)
    dictioanry = UnicodeCharsVocabulary(words, len(examples),
                                        args.max_characters_per_token)
    return dictioanry


def top_words(args, examples, word_dict):
    """Count and return the most common question words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['question'].word:
            w = Vocabulary.normalize(w)
            if w in word_dict:
                word_count.update([w])
    return word_count.most_common(args.tune_partial)
