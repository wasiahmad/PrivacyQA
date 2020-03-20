import copy
import numpy as np
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

from nqa.config import override_model_args, add_new_model_args
from nqa.models.bidaf import BIDAF
from nqa.modules.embeddings import Embeddings
from nqa.utils.misc import count_file_lines
from nqa.inputters.constants import PAD, PAD_WORD

logger = logging.getLogger(__name__)


class QAModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, src_dict, state_dict=None):
        # Book-keeping.
        self.name = 'BiDAF'
        self.args = args
        self.src_dict = src_dict
        self.args.vocab_size = len(src_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        self.args.extra_feat = 0
        if self.args.use_tf:
            self.args.extra_feat += 1
        if self.args.use_idf:
            self.args.extra_feat += 1

        self.network = BIDAF(self.args)
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, self.args.pos_weight]))

        # Load saved state
        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)

    def load_embeddings(self, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.
        #TODO: update args
        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in self.src_dict.tokens()}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts, embedding = {}, {}
        with open(embedding_file) as f:
            # Skip first line if of form count/dim.
            line = f.readline().rstrip().split(' ')
            if len(line) != 2:
                f.seek(0)

            duplicates = set()
            for line in tqdm(f, total=count_file_lines(embedding_file)):
                parsed = line.rstrip().split(' ')
                assert len(parsed) == self.args.emsize + 1
                w = self.src_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[w] = vec
                    else:
                        duplicates.add(w)
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[w].add_(vec)

            if len(duplicates) > 0:
                logging.warning(
                    'WARN: Duplicate embedding found for %s' % ', '.join(duplicates)
                )

        for w, c in vec_counts.items():
            embedding[w].div_(c)

        self.network.embedder.word_embeddings.init_word_vectors(self.src_dict, embedding)
        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def tune_embeddings(self, words):
        """Unfix the embeddings of a list of words. This is only relevant if
        only some of the embeddings are being tuned (tune_partial = N).
        Shuffles the N specified words to the front of the dictionary, and saves
        the original vectors of the other N + 1:vocab words in a fixed buffer.
        Args:
            words: iterable of tokens contained in dictionary.
        """
        words = {w for w in words if w in self.src_dict}

        if len(words) == 0:
            logger.warning('Tried to tune embeddings, but no words given!')
            return

        if len(words) == len(self.src_dict):
            logger.warning('Tuning ALL embeddings in dictionary')
            return

        # Shuffle words and vectors
        embedding = self.network.embedder.word_embeddings.word_lut.weight.data
        for idx, swap_word in enumerate(words, self.src_dict.START):
            # Get current word + embedding for this index
            curr_word = self.src_dict[idx]
            curr_emb = embedding[idx].clone()
            old_idx = self.src_dict[swap_word]

            # Swap embeddings + dictionary indices
            embedding[idx].copy_(embedding[old_idx])
            embedding[old_idx].copy_(curr_emb)
            self.src_dict[swap_word] = idx
            self.src_dict[idx] = swap_word
            self.src_dict[curr_word] = old_idx
            self.src_dict[old_idx] = curr_word

        # Save the original, fixed embeddings
        self.network.register_buffer(
            'fixed_embedding', embedding[idx + 1:].clone()
        )

    def init_optimizer(self, state_dict=None, use_gpu=True):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: optimizer's state dict
            use_gpu: required to move state_dict to GPU
        """
        if self.args.fix_embeddings:
            self.network.embedder.word_embeddings.fix_word_lut()

        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(parameters, self.args.learning_rate,
                                        weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.args.optimizer)

        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)
            # FIXME: temp soln - https://github.com/pytorch/pytorch/issues/2830
            if use_gpu:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        sentence_word_rep = ex['sentence_word_rep']
        sentence_char_rep = ex['sentence_char_rep']
        sentence_len = ex['sentence_len']
        question_word_rep = ex['question_word_rep']
        question_char_rep = ex['question_char_rep']
        question_len = ex['question_len']
        label = ex['label']

        if self.use_cuda:
            sentence_len = sentence_len.cuda(non_blocking=True)
            question_len = question_len.cuda(non_blocking=True)
            sentence_word_rep = sentence_word_rep.cuda(non_blocking=True)
            question_word_rep = question_word_rep.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            sentence_char_rep = sentence_char_rep.cuda(non_blocking=True) \
                if sentence_char_rep is not None else None
            question_char_rep = question_char_rep.cuda(non_blocking=True) \
                if question_char_rep is not None else None

        # Run forward
        score = self.network(sentence_word_rep=sentence_word_rep,
                             sentence_char_rep=sentence_char_rep,
                             sentence_len=sentence_len,
                             question_word_rep=question_word_rep,
                             question_char_rep=question_char_rep,
                             question_len=question_len)

        # Compute loss and accuracies
        loss = self.criterion(score, label)

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                       self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

        return loss.item()

    def reset_parameters(self):
        """Reset any partially fixed parameters to original states."""

        # Reset fixed embeddings to original value
        if self.args.tune_partial > 0:
            if self.parallel:
                embedding = self.network.module.embedder.word_embeddings.word_lut.weight.data
                fixed_embedding = self.network.module.fixed_embedding
            else:
                embedding = self.network.embedder.word_embeddings.word_lut.weight.data
                fixed_embedding = self.network.fixed_embedding

            # Embeddings to fix are the last indices
            offset = embedding.size(0) - fixed_embedding.size(0)
            if offset >= 0:
                embedding[offset:] = fixed_embedding

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex):
        """Forward a batch of examples only to get predictions.
        Args:
            ex: the batch
        Output:
            loss: batch * top_n predicted start indices
            probs: batch * top_n predicted end indices
            predictions: batch * top_n prediction scores
        """
        # Eval mode
        self.network.eval()

        sentence_word_rep = ex['sentence_word_rep']
        sentence_char_rep = ex['sentence_char_rep']
        sentence_len = ex['sentence_len']
        question_word_rep = ex['question_word_rep']
        question_char_rep = ex['question_char_rep']
        question_len = ex['question_len']
        label = ex['label']

        if self.use_cuda:
            sentence_len = sentence_len.cuda(non_blocking=True)
            question_len = question_len.cuda(non_blocking=True)
            sentence_word_rep = sentence_word_rep.cuda(non_blocking=True)
            question_word_rep = question_word_rep.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            sentence_char_rep = sentence_char_rep.cuda(non_blocking=True) \
                if sentence_char_rep is not None else None
            question_char_rep = question_char_rep.cuda(non_blocking=True) \
                if question_char_rep is not None else None

        score = self.network(sentence_word_rep=sentence_word_rep,
                             sentence_char_rep=sentence_char_rep,
                             sentence_len=sentence_len,
                             question_word_rep=question_word_rep,
                             question_char_rep=question_char_rep,
                             question_len=question_len)

        loss = self.criterion(score, label)
        probs = f.softmax(score, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(score.data.cpu().numpy(), axis=1).tolist()

        return {
            'loss': loss,
            'probs': probs,
            'predictions': predictions,
        }

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'src_dict': self.src_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        params = {
            'state_dict': network.state_dict(),
            'src_dict': self.src_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        src_dict = saved_params['src_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
            args = add_new_model_args(args, new_args)
        return QAModel(args, src_dict, state_dict)

    @staticmethod
    def load_checkpoint(filename, use_gpu=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        src_dict = saved_params['src_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = QAModel(args, src_dict, state_dict)
        model.init_optimizer(optimizer, use_gpu)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()
        self.criterion = self.criterion.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()
        self.criterion = self.criterion.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
