# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/data.py
import unicodedata
import numpy as np
from nqa.inputters import constants


class Vocabulary(object):

    def __init__(self, type='regular'):
        self.tok2ind = dict()
        self.ind2tok = dict()
        self.tok2idf = dict()

        if type.lower() == 'bert':
            self._pad_word = constants.BERT_PAD_WORD
            self._unk_word = constants.BERT_UNK_WORD
        elif type.lower() == 'regular':
            self._pad_word = constants.PAD_WORD
            self._unk_word = constants.UNK_WORD
            self.tok2ind[constants.PAD_WORD] = constants.PAD
            self.tok2ind[constants.UNK_WORD] = constants.UNK
            self.ind2tok[constants.PAD] = constants.PAD_WORD
            self.ind2tok[constants.UNK] = constants.UNK_WORD
        else:
            raise NotImplementedError

    @property
    def pad_word(self):
        return self._pad_word

    @property
    def unk_word(self):
        return self._unk_word

    @property
    def pad_idx(self):
        return self.tok2ind[self._pad_word]

    @property
    def unk_idx(self):
        return self.tok2ind[self._unk_word]

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.unk_word)
        elif type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.unk_word))
        else:
            raise RuntimeError('Invalid key type.')

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token, idf=0):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token
            self.tok2idf[token] = idf

    def add_tokens(self, token_list):
        assert isinstance(token_list, list)
        for token in token_list:
            self.add(token)

    def add_sepcial_tokens(self):
        self.add(self.pad_word)
        self.add(self.unk_word)

    def tokens(self):
        """Get dictionary tokens.
        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {self.pad_word, self.unk_word}]
        return tokens

    def load(self, vocab):
        for token, index in vocab.items():
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def load_df(self, inv_doc_freq):
        for token, idf in inv_doc_freq.items():
            self.tok2idf[token] = idf

    def remove(self, key):
        if key in self.tok2ind:
            ind = self.tok2ind[key]
            del self.tok2ind[key]
            del self.ind2tok[ind]
            return True
        return False


class UnicodeCharsVocabulary(Vocabulary):
    """Vocabulary containing character-level and word level information.
    Has a word vocabulary that is used to lookup word ids and
    a character id that is used to map words to arrays of character ids.
    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.
    """

    def __init__(self, words, num_example, max_word_length, df=None):
        super(UnicodeCharsVocabulary, self).__init__()
        self._max_word_length = max_word_length

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars
        self.bow_char = 256  # <begin word>
        self.eow_char = 257  # <end word>
        self.pad_char = 258  # <padding>

        for w in words:
            if df is not None:
                self.add(w, np.log(1.0 * num_example / df[w]))
            else:
                self.add(w)
        num_words = len(self.ind2tok)

        self._word_char_ids = np.zeros([num_words, max_word_length],
                                       dtype=np.int32)

        for i, word in self.ind2tok.items():
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = self.pad_char

        word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length - 2)]
        code[0] = self.bow_char
        k = 0
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[k + 1] = self.eow_char

        return code

    def word_to_char_ids(self, word):
        if word in self.tok2ind:
            return self._word_char_ids[self.tok2ind[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence, split=True):
        '''
        Encode the sentence as a white space delimited string of tokens.
        '''
        if split:
            chars_ids = [self.word_to_char_ids(cur_word)
                         for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word)
                         for cur_word in sentence]

        return chars_ids
