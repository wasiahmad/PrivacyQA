class Sentence(object):

    def __init__(self, _id=None):
        self._id = _id
        self._word = []
        self._lemma = []
        self._pos = []
        self._ner = []
        self._head = []
        self._deprel = []
        self._spaceafter = []

    @property
    def id(self) -> str:
        return self._id

    @property
    def word(self) -> list:
        return self._word

    @property
    def text(self) -> str:
        s_text = ''
        for word, space in zip(self._word, self._spaceafter):
            s_text += word
            s_text += ' ' if space else ''
        return s_text

    @property
    def lemma(self) -> list:
        return self._lemma

    @property
    def pos(self) -> list:
        return self._pos

    @property
    def ner(self) -> list:
        return self._ner

    @property
    def head(self) -> list:
        return self._head

    @property
    def deprel(self) -> list:
        return self._deprel

    @property
    def spaceafter(self) -> list:
        return self._spaceafter

    @word.setter
    def word(self, param: list) -> None:
        assert isinstance(param, list)
        self._word = param

    @lemma.setter
    def lemma(self, param: list) -> None:
        assert isinstance(param, list)
        self._lemma = param

    @pos.setter
    def pos(self, param: list) -> None:
        assert isinstance(param, list)
        self._pos = param

    @ner.setter
    def ner(self, param: list) -> None:
        assert isinstance(param, list)
        self._ner = param

    @head.setter
    def head(self, param: list) -> None:
        assert isinstance(param, list)
        self._head = param

    @deprel.setter
    def deprel(self, param: list) -> None:
        assert isinstance(param, list)
        self._deprel = param

    @spaceafter.setter
    def spaceafter(self, param: list) -> None:
        assert isinstance(param, list)
        self._spaceafter = param

    def __len__(self):
        return len(self._word)
