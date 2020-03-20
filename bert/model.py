import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

from nqa.config import override_model_args, add_new_model_args
from nqa.models import BertQA
from nqa.transformers import BertConfig, BertModel, WarmupLinearSchedule, AdamW
from nqa.inputters import constants

logger = logging.getLogger(__name__)

logger.info('FP16 activate, use apex FusedAdam')
try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


class BertModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, vocab, tokenizer, **kwargs):
        # Book-keeping.
        self.name = 'BertQA'
        self.args = args
        self.vocab = vocab
        self.updates = 0
        self.use_cuda = False
        self.parallel = False
        self.tokenizer = tokenizer

        if 'state_dict' in kwargs:
            config = kwargs['config']
            self.updates = kwargs.get('updates', 0)
            self.network = BertQA(config)
            self.network.load_state_dict(kwargs['state_dict'])

        elif self.args.bert_model is not None:
            config = BertConfig.from_pretrained(self.args.bert_config_file)
            config.dropout = self.args.dropout
            self.network = BertQA(config)
            bert_states = torch.load(self.args.bert_weight_file)
            self.network.load_pretrained_weights(bert_states)
            logger.info('Bert weights are loaded from %s' % self.args.bert_weight_file)

        else:
            raise NotImplementedError

        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, self.args.pos_weight]))
        self.optimizer = None
        self.scheduler = None
        self.config = config

    def activate_fp16(self):
        if self.args.fp16:
            # https://github.com/NVIDIA/apex/issues/227
            assert self.optimizer is not None
            self.network, self.optimizer = amp.initialize(self.network,
                                                          self.optimizer,
                                                          opt_level=self.args.fp16_opt_level)

    def init_optimizer(self, state_dict=None, use_gpu=True):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: optimizer's state dict
            use_gpu: required to move state_dict to GPU
        """
        param_optimizer = list(self.network.named_parameters())

        # There seems to be something that we can't
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        num_train_optimization_steps = int(
            self.args.num_train_examples / self.args.batch_size / self.args.gradient_accumulation_steps) \
                                       * self.args.num_epochs

        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=self.args.learning_rate)
        self.scheduler = WarmupLinearSchedule(self.optimizer,
                                              warmup_steps=self.args.warmup_steps,
                                              t_total=num_train_optimization_steps)

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

        source_ids = ex['source_ids']
        source_pos_ids = ex['source_pos_ids']
        source_type_ids = ex['source_type_ids']
        source_mask = ex['source_mask']
        label = ex['label']

        if self.use_cuda:
            label = label.cuda(non_blocking=True)
            source_ids = source_ids.cuda(non_blocking=True)
            source_pos_ids = source_pos_ids.cuda(non_blocking=True) \
                if source_pos_ids is not None else None
            source_type_ids = source_type_ids.cuda(non_blocking=True) \
                if source_type_ids is not None else None
            source_mask = source_mask.cuda(non_blocking=True) \
                if source_mask is not None else None

        # Run forward
        score = self.network(source_ids=source_ids,
                             source_pos_ids=source_pos_ids,
                             source_type_ids=source_type_ids,
                             source_mask=source_mask)

        # Compute loss and accuracies
        loss = self.criterion(score, label)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if (self.updates + 1) % self.args.gradient_accumulation_steps == 0:
            if self.args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.grad_clipping)
            else:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.grad_clipping)

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.optimizer.zero_grad()

        self.updates += 1

        return loss.item()

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex):
        """Forward a batch of examples only to get predictions.
        Args:
            ex: the batch examples
        Output:
            predictions: #batch predicted sequences
        """
        # Eval mode
        self.network.eval()

        source_ids = ex['source_ids']
        source_pos_ids = ex['source_pos_ids']
        source_type_ids = ex['source_type_ids']
        source_mask = ex['source_mask']
        label = ex['label']

        if self.use_cuda:
            label = label.cuda(non_blocking=True)
            source_ids = source_ids.cuda(non_blocking=True)
            source_pos_ids = source_pos_ids.cuda(non_blocking=True) \
                if source_pos_ids is not None else None
            source_type_ids = source_type_ids.cuda(non_blocking=True) \
                if source_type_ids is not None else None
            source_mask = source_mask.cuda(non_blocking=True) \
                if source_mask is not None else None

        score = self.network(source_ids=source_ids,
                             source_pos_ids=source_pos_ids,
                             source_type_ids=source_type_ids,
                             source_mask=source_mask)

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
        params = {
            'config': self.config,
            'state_dict': state_dict,
            'vocab': self.vocab,
            'tokenizer': self.tokenizer,
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
            'config': self.config,
            'state_dict': network.state_dict(),
            'vocab': self.vocab,
            'tokenizer': self.tokenizer,
            'args': self.args,
            'epoch': epoch,
            'updates': self.updates,
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
        vocab = saved_params['vocab']
        tokenizer = saved_params['tokenizer']
        state_dict = saved_params['state_dict']
        config = saved_params['config']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
            args = add_new_model_args(args, new_args)
        return BertModel(args, vocab, tokenizer,
                         state_dict=state_dict,
                         config=config)

    @staticmethod
    def load_checkpoint(filename, use_gpu=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        vocab = saved_params['vocab']
        tokenizer = saved_params['tokenizer']
        state_dict = saved_params['state_dict']
        config = saved_params['config']
        epoch = saved_params['epoch']
        updates = saved_params['updates']
        optimizer_states = saved_params['optimizer']
        args = saved_params['args']
        model = BertModel(args, vocab, tokenizer,
                          state_dict=state_dict,
                          config=config,
                          updates=updates)
        model.init_optimizer(optimizer_states, use_gpu)
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
