# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/data.py
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from bidaf.vector import vectorize as bidaf_vectorize
from bert.vector import vectorize as bert_vectorize


# ------------------------------------------------------------------------------
# PyTorch dataset class for SQuAD (and SQuAD-like) data.
# ------------------------------------------------------------------------------


class ReaderDataset(Dataset):
    def __init__(self, examples, model):
        self.model = model
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        """Generates one sample of data"""
        if self.model.name == 'BertQA':
            return bert_vectorize(self.examples[index], self.model)
        elif self.model.name == 'BiDAF':
            return bidaf_vectorize(self.examples[index], self.model)
        else:
            raise NotImplementedError

    def lengths(self):
        return [(len(ex['sentence']), len(ex['question']))
                for ex in self.examples]

    def labels(self):
        return [ex['one_label'] for ex in self.examples]


# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and question).
# ------------------------------------------------------------------------------


class SortedBatchSampler(Sampler):
    def __init__(self, lengths, labels, batch_size,
                 shuffle=True, filter=True):
        self.lengths = lengths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filter = filter

    def indices(self):
        pos_indices = [i for i, l in enumerate(self.labels) if l == 1]
        neg_indices = [i for i, l in enumerate(self.labels) if l == 0]
        if len(neg_indices) > len(pos_indices):
            np.random.shuffle(neg_indices)
            neg_indices = neg_indices[:len(pos_indices)]
        return pos_indices + neg_indices

    def __iter__(self):
        indices = self.indices() if self.filter else list(range(len(self.lengths)))
        sorted_indices = np.array(
            [(-self.lengths[idx][0], -self.lengths[idx][1], np.random.random())
             for idx in indices],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
        )

        sorted_indices = np.argsort(sorted_indices, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(sorted_indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)
