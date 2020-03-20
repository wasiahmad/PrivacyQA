import os
import sys

sys.path.append('../..')
import string
from tqdm import tqdm
from collections import OrderedDict
from nqa.tokenizers import SpacyTokenizer

simple_tokenizer = SpacyTokenizer()
tokenizer = SpacyTokenizer(annotators={'pos', 'lemma', 'ner'})

FIELD_MAP = {
    0: 'Folder',
    1: 'DocId',
    2: 'QueryId',
    3: 'SentId',
    4: 'Split',
    5: 'Query',
    6: 'Segment',
    7: 'Label'
}

OPP_MAP = {
    5: 'first',
    6: 'third',
    7: 'datasecurity',
    8: 'dataretention',
    9: 'user_access',
    10: 'user_choice',
    11: 'other'
}


class Document:
    def __init__(self, id):
        self._id = id
        self._sentences = dict()
        self._questions = dict()

    @property
    def id(self):
        return self._id

    @property
    def sentences(self):
        sorted_x = sorted(self._sentences.items(), key=lambda k: k[0])
        return [item[1] for item in sorted_x]

    @sentences.setter
    def sentences(self, param):
        assert isinstance(param, tuple)
        sid, stext = param
        if sid in self._sentences:
            assert stext == self._sentences[sid]
        else:
            self._sentences[sid] = stext

    @property
    def questions(self):
        sorted_x = sorted(self._questions.items(), key=lambda k: k[0])
        return [item[1] for item in sorted_x]

    @questions.setter
    def questions(self, param):
        assert len(param) == 3
        qid, qtext, qlabel = param
        if qid in self._questions:
            assert qtext == self._questions[qid]['text']
        else:
            self._questions[qid] = {'text': qtext,
                                    'relevant': dict(),
                                    'type': list()}
        if len(qlabel) > 0:
            assert isinstance(qlabel, dict)
            for anno_id, anno in qlabel.items():
                if anno_id not in self._questions[qid]['relevant']:
                    self._questions[qid]['relevant'][anno_id] = list()
                self._questions[qid]['relevant'][anno_id].extend(anno)

    def set_question_type(self, param):
        assert len(param) == 3
        qid, qtext, qtype = param
        assert qid in self._questions
        assert qtext == self._questions[qid]['text']
        assert isinstance(qtype, list)
        self._questions[qid]['type'] = qtype


def main():
    vocab = set()

    def read_documents(mainfile, metafile, is_test=False):
        with open(mainfile) as f:
            next(f)
            documents = OrderedDict()
            for line in f:
                fields = line.strip().split('\t')
                if fields[1] not in documents:
                    documents[fields[1]] = Document(fields[1])

                # we add 1 because original_sent_ids are 0' based
                sent_id = int(fields[3].split('_')[-1]) + 1
                documents[fields[1]].sentences = (sent_id, fields[6])

                query_id = int(fields[2].split('_')[-1])
                qlabel = dict()
                if is_test:
                    # test split
                    for j, val in enumerate(fields[-6:]):
                        assert val in ['Relevant', 'Irrelevant', 'None']
                        if val != 'None':
                            qlabel[j] = []
                        if val == 'Relevant':
                            qlabel[j].append(sent_id)
                else:
                    # train split
                    assert len(fields) == len(FIELD_MAP)
                    if fields[7] != 'None':
                        qlabel[0] = []
                    if fields[7] == 'Relevant':
                        qlabel[0].append(sent_id)

                documents[fields[1]].questions = (query_id, fields[5], qlabel)

        with open(metafile) as f:
            next(f)
            for line in f:
                fields = line.strip().split('\t')
                doc_id = fields[1]
                query_id = int(fields[2].split('_')[-1])
                qtext = fields[4]

                qlabel = []
                for j in range(5, 12):
                    if int(fields[j]) == 1:
                        qlabel.append(OPP_MAP[j])

                documents[doc_id].set_question_type((query_id, qtext, qlabel))

        return list(documents.values())

    def save_documents(documents, outdir):
        for idx, doc in enumerate(tqdm(documents, total=len(documents))):
            com_name = doc.id
            com_name = com_name.translate(str.maketrans('', '', string.punctuation))
            com_name = '_'.join(com_name.split()[:-1]).lower()

            # practices = read_annotations(cname)
            sentence_idx = 0
            with open(outdir + com_name + '.sentence', 'w') as f1:
                for sent_id, sentence in enumerate(doc.sentences):
                    sentence_idx += 1

                    s_toks = tokenizer.tokenize(sentence.strip())
                    words = s_toks.words()
                    vocab.update(words)
                    pos_tags = s_toks.pos()
                    ners = s_toks.entities()
                    lemmas = s_toks.lemmas()
                    safters = s_toks.space_after()
                    dep_rels = s_toks.dep_rels()
                    assert len(words) == len(pos_tags)

                    offset = 0
                    noun_chunks = s_toks.noun_chunks

                    chunk_annot = [0] * len(words)
                    if noun_chunks:
                        for (start, end) in noun_chunks:
                            chunk_annot[start:end] = [1] * (end - start)

                    for idx in range(len(words)):
                        ner_tag = ners[idx] if ners[idx] else '_'
                        SpaceAfter = 'SpaceAfter=No' if safters[idx] else '_'
                        drel = '%d:%s' % (dep_rels[idx][0] + 1 - offset, dep_rels[idx][2]) \
                            if dep_rels[idx][2] else '_'
                        f1.write('%d\t%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\n' %
                                 (sentence_idx, idx + 1,
                                  words[idx], lemmas[idx], pos_tags[idx],
                                  ner_tag, drel, chunk_annot[idx],
                                  SpaceAfter))

                    f1.write('\n')
                    offset += len(words)

            with open(outdir + com_name + '.question', 'w') as f2, \
                    open(outdir + com_name + '.label', 'w') as f3:
                for qid, question in enumerate(doc.questions):
                    s_toks = tokenizer.tokenize(question['text'].strip())
                    words = s_toks.words()
                    vocab.update(words)
                    pos_tags = s_toks.pos()
                    ners = s_toks.entities()
                    lemmas = s_toks.lemmas()
                    safters = s_toks.space_after()
                    dep_rels = s_toks.dep_rels()
                    assert len(words) == len(pos_tags)

                    offset = 0
                    noun_chunks = s_toks.noun_chunks

                    chunk_annot = [0] * len(words)
                    if noun_chunks:
                        for (start, end) in noun_chunks:
                            chunk_annot[start:end] = [1] * (end - start)

                    for idx in range(len(words)):
                        ner_tag = ners[idx] if ners[idx] else '_'
                        SpaceAfter = 'SpaceAfter=No' if safters[idx] else '_'
                        drel = '%d:%s' % (dep_rels[idx][0] + 1 - offset, dep_rels[idx][2]) \
                            if dep_rels[idx][2] else '_'
                        f2.write('%d\t%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\n' %
                                 (qid + 1, idx, words[idx], lemmas[idx], pos_tags[idx],
                                  ner_tag, drel, chunk_annot[idx],
                                  SpaceAfter))

                    f2.write('\n')
                    offset += len(words)

                    # a list of relevant sentences
                    # question['relevant'] is a dict
                    rel_labels = []
                    for annotations in question['relevant'].values():
                        annotations = list(annotations)
                        if annotations:
                            annotations.sort()
                            annotations = ','.join([str(v) for v in annotations])
                        else:
                            annotations = '_'
                        rel_labels.append(annotations)

                    qtype = question['type']
                    if qtype:
                        qtype = '||'.join(qtype)
                    else:
                        qtype = '_'

                    rel_labels = '\t'.join(rel_labels)
                    f3.write('%d\t%s\t%s' % (qid + 1, qtype, rel_labels) + '\n')

    outdir = './train/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    train_docs = read_documents('policy_train_data.csv',
                                'train_opp_annotations.csv',
                                is_test=False)
    assert len(train_docs) == 27
    save_documents(train_docs[:24], outdir)

    outdir = './valid/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    save_documents(train_docs[24:], outdir)

    outdir = './test/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    test_docs = read_documents('policy_test_data.csv',
                               'test_opp_annotations.csv',
                               is_test=True)
    assert len(test_docs) == 8
    save_documents(test_docs, outdir)

    with open('vocab.txt', 'w') as fw:
        for word in vocab:
            fw.write(word + '\n')


if __name__ == '__main__':
    main()
