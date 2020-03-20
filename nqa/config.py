""" Implementation of all available options """
from __future__ import print_function

"""Model architecture/optimization options for Seq2seq architecture."""

import argparse
import logging

logger = logging.getLogger(__name__)

# Index of arguments concerning the core model architecture
MODEL_ARCHITECTURE = {
    'model_type',
    'emsize',
    'rnn_type',
    'nhid',
    'nlayers',
    'bidirection',
    'use_word',
    'use_chars',
    'n_characters',
    'char_emsize',
    'filter_size',
    'nfilters',
    'use_pos',
    'd_ff',
    'num_head',
    'trans_drop'
}

BERT_CONFIG = {
    'attention_probs_dropout_prob',
    'hidden_act',
    'hidden_dropout_prob',
    'hidden_size',
    'initializer_range',
    'intermediate_size',
    'max_position_embeddings',
    'num_attention_heads',
    'num_hidden_layers',
    'type_vocab_size',
    'vocab_size'
}

DATA_OPTIONS = {
    'max_sent_len',
    'use_tf',
    'use_idf'
}

ADVANCED_OPTIONS = {
    'bert_model',
    'bert_weight_file',
    'bert_config_file',
    'parallel',
    'num_train_examples',
    'batch_size',
    'num_epochs'
}

# Index of arguments concerning the model optimizer/training
MODEL_OPTIMIZER = {
    'fix_embeddings',
    'optimizer',
    'learning_rate',
    'momentum',
    'weight_decay',
    'dropout_rnn',
    'dropout',
    'dropout_emb',
    'cuda',
    'grad_clipping',
    'tune_partial',
    'lr_decay',
    'use_warmup_schedule',
    'gradient_accumulation_steps',
    'warmup_steps',
    'fp16',
    'fp16_opt_level',
    'loss_scale',
    'pos_weight'
}


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_model_args(parser):
    parser.register('type', 'bool', str2bool)

    # Data options
    data = parser.add_argument_group('Data parameters')
    data.add_argument('--max_sent_len', type=int, default=100,
                      help='Maximum allowed length for a sentence')
    data.add_argument('--use_tf', type='bool', default=False,
                      help='Use term frequency as additional feature')
    data.add_argument('--use_idf', type='bool', default=False,
                      help='Use inverted document frequency as additional feature')

    # Model architecture
    model = parser.add_argument_group('Keyphrase Generator')
    model.add_argument('--model_type', type=str, default='rnn',
                       help='Model architecture type')
    model.add_argument('--emsize', type=int, default=300,
                       help='Embedding size if embedding_file is not given')
    model.add_argument('--rnn_type', type=str, default='LSTM',
                       help='RNN type: LSTM, GRU')
    model.add_argument('--nhid', type=int, default=200,
                       help='Hidden size of RNN units')
    model.add_argument('--bidirection', type='bool', default=True,
                       help='use bidirectional recurrent unit')
    model.add_argument('--nlayers', type=int, default=2,
                       help='Number of encoding layers')

    # Transformer specific params
    model.add_argument('--use_pos', type='bool', default=True,
                       help='Use position embeddings')
    model.add_argument('--d_ff', type=int, default=2048,
                       help='Number of units in position-wise FFNN')
    model.add_argument('--num_head', type=int, default=8,
                       help='Number of heads in Multi-Head Attention')
    model.add_argument('--trans_drop', type=float, default=0.2,
                       help='Dropout for transformer')

    # Input representation specific details
    model.add_argument('--use_word', type='bool', default=True,
                       help='Use character embedding in the input')
    model.add_argument('--use_chars', type='bool', default=False,
                       help='Use character embedding in the input')
    model.add_argument('--n_characters', type=int, default=260,
                       help='Character vocabulary size')
    model.add_argument('--char_emsize', type=int, default=16,
                       help='Character embedding size')
    model.add_argument('--filter_size', nargs='+', type=int,
                       help='Char convolution filter sizes')
    model.add_argument('--nfilters', nargs='+', type=int,
                       help='Number of char convolution filters')

    # Optimization details
    optim = parser.add_argument_group('Neural QA Reader Optimization')
    optim.add_argument('--dropout_emb', type=float, default=0.2,
                       help='Dropout rate for word embeddings')
    optim.add_argument('--dropout_rnn', type=float, default=0.2,
                       help='Dropout rate for RNN states')
    optim.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout for NN layers')
    optim.add_argument('--optimizer', type=str, default='adam',
                       help='Optimizer: sgd or adamax')
    optim.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for the optimizer')
    parser.add_argument('--lr_decay', type=float, default=0.95,
                        help='Decay ratio for learning rate')
    optim.add_argument('--grad_clipping', type=float, default=5.0,
                       help='Gradient clipping')
    parser.add_argument('--early_stop', type=int, default=5,
                        help='Stop training if performance doesn\'t improve')
    optim.add_argument('--weight_decay', type=float, default=0,
                       help='Weight decay factor')
    optim.add_argument('--momentum', type=float, default=0,
                       help='Momentum factor')
    optim.add_argument('--fix_embeddings', type='bool', default=True,
                       help='Keep word embeddings fixed (use pretrained)')
    optim.add_argument('--tune_partial', type=int, default=0,
                       help='Backprop through only the top N question words')
    optim.add_argument('--use_warmup_schedule', type='bool', default=False,
                       help='Use warmup lrate schedule')
    optim.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of steps for gradient accumulation')
    optim.add_argument('--warmup_steps', type=int, default=10000,
                       help='Percentage of warmup proportion')
    optim.add_argument('--pos_weight', type=float, default=1.0,
                       help='Weight of the positive examples in the loss function')

    bert = parser.add_argument_group('Bert Configurations')
    bert.add_argument('--bert_model', type=str, default='bert_base_uncased',
                      help='Model name of the BERT')
    bert.add_argument('--attention_probs_dropout_prob', type=float, default=0.1,
                      help='Dropout rate for word embeddings')
    bert.add_argument('--hidden_act', type=str, default='gelu',
                      help='Hidden activation function')
    bert.add_argument('--hidden_dropout_prob', type=float, default=0.1,
                      help='Dropout for hidden layers')
    bert.add_argument('--hidden_size', type=int, default=768,
                      help='Hidden size of sublayers')
    bert.add_argument('--initializer_range', type=float, default=0.02,
                      help='Initializer range for weight initialization')
    bert.add_argument('--intermediate_size', type=int, default=3072,
                      help='Intermediate size of position-wise feed-forward layer')
    bert.add_argument('--max_position_embeddings', type=int, default=512,
                      help='Maximum length of the input')
    bert.add_argument('--num_attention_heads', type=int, default=12,
                      help='Number of attention heads for multi-head attention')
    bert.add_argument('--num_hidden_layers', type=int, default=12,
                      help='Number of hidden layers in encoder and decoder')
    bert.add_argument('--type_vocab_size', type=int, default=2,
                      help='Size of the BERT type vocabulary')
    bert.add_argument('--vocab_size', type=int, default=30522,
                      help='Size of the BERT vocabulary')


def get_model_args(args):
    """Filter args for model ones.
    From a args Namespace, return a new Namespace with *only* the args specific
    to the model architecture or optimization. (i.e. the ones defined here.)
    """
    global MODEL_ARCHITECTURE, MODEL_OPTIMIZER, ADVANCED_OPTIONS, \
        DATA_OPTIONS, BERT_CONFIG

    required_args = MODEL_ARCHITECTURE | MODEL_OPTIMIZER | ADVANCED_OPTIONS \
                    | DATA_OPTIONS | BERT_CONFIG

    arg_values = {k: v for k, v in vars(args).items() if k in required_args}
    return argparse.Namespace(**arg_values)


def override_model_args(old_args, new_args):
    """Set args to new parameters.
    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.
    We keep the new optimization or RL setting, and leave the model architecture alone.
    """
    global MODEL_OPTIMIZER
    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k]:
            if k in MODEL_OPTIMIZER:
                logger.info('Overriding saved %s: %s --> %s' %
                            (k, old_args[k], new_args[k]))
                old_args[k] = new_args[k]
            else:
                logger.info('Keeping saved %s: %s' % (k, old_args[k]))

    return argparse.Namespace(**old_args)


def add_new_model_args(old_args, new_args):
    """Set args to new parameters.
    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.
    We keep the new optimization or RL setting, and leave the model architecture alone.
    """
    global ADVANCED_OPTIONS
    old_args, new_args = vars(old_args), vars(new_args)
    for k in new_args.keys():
        if k not in old_args:
            if (k in ADVANCED_OPTIONS):
                logger.info('Adding arg %s: %s' % (k, new_args[k]))
                old_args[k] = new_args[k]

    return argparse.Namespace(**old_args)
