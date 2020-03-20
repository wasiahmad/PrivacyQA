import spacy
import copy
from .tokenizer import Tokens, Tokenizer


class SpacyTokenizer(Tokenizer):

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """
        model = kwargs.get('model', 'en')
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        nlp_kwargs = {'parser': False}
        if not any([p in self.annotators for p in ['lemma', 'pos', 'ner']]):
            nlp_kwargs['tagger'] = False
        if 'ner' not in self.annotators:
            nlp_kwargs['entity'] = False
        self.nlp = spacy.load(model, **nlp_kwargs)

    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')
        doc = self.nlp(clean_text)
        if any([p in self.annotators for p in ['lemma', 'pos', 'ner']]):
            self.nlp.tagger(doc)
        if 'ner' in self.annotators:
            self.nlp.entity(doc)

        data = []
        for i in range(len(doc)):
            # Get whitespace
            start_ws = doc[i].idx
            if i + 1 < len(doc):
                end_ws = doc[i + 1].idx
            else:
                end_ws = doc[i].idx + len(doc[i].text)

            data.append((
                doc[i].text,
                text[start_ws: end_ws],
                (doc[i].idx, doc[i].idx + len(doc[i].text)),
                doc[i].tag_,
                doc[i].lemma_,
                doc[i].ent_type_,
                (doc[i].head.i, doc[i].head.text, doc[i].dep_)
            ))

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        tokens = Tokens(data, self.annotators, opts={'non_ent': ''})
        tokens.sent_boundaries = [[sent.start, sent.end] for sent in doc.sents]
        tokens.noun_chunks = [[np.start, np.end] for np in doc.noun_chunks]
        return tokens
