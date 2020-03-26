# src: https://github.com/facebookresearch/DrQA/blob/master/scripts/reader/train.py

import sys

sys.path.append(".")
sys.path.append("..")

import os
import math
import json
import torch
import logging
import subprocess
import argparse
import numpy as np

import nqa.config as config
import nqa.inputters as util

from collections import OrderedDict
from tqdm import tqdm
from nqa.utils.timer import AverageMeter, Timer
import bidaf.vector as vector
import nqa.inputters.dataset as data
from nqa.inputters import UnicodeCharsVocabulary

from bidaf.model import QAModel
from nqa.utils import scorer

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--data_workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--random_seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num_epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch_size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test_batch_size', type=int, default=128,
                         help='Batch size during validation/testing')
    runtime.add_argument('--filter', type='bool', default=False,
                         help='Filter training data to balance positive/negative examples')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--dataset_name', type=str, default='privacyQA',
                       choices=['privacyQA'],
                       help='Name of the experimental dataset')
    files.add_argument('--model_dir', type=str, default='/tmp/',
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model_name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data_dir', type=str, default='/data/privacyQA/',
                       help='Directory of training/validation data')
    files.add_argument('--train_dir', type=str, default='train/',
                       help='Preprocessed train file')
    files.add_argument('--valid_dir', type=str, default='valid/',
                       help='Preprocessed dev file')
    files.add_argument('--vocab_file', type=str, default='',
                       help='Preprocessed vocab file')
    files.add_argument('--embed_dir', type=str, default='/data/fasttext/',
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding_file', type=str, default='',
                       help='Space-separated pretrained embeddings file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default=None,
                           help='Path to a pretrained model to warm-start with')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--max_examples', type=int, default=-1,
                            help='Maximum number of examples for training')
    preprocess.add_argument('--restrict_vocab', type='bool', default=True,
                            help='Only use pre-trained words in embedding_file')
    preprocess.add_argument('--src_vocab_size', type=int, default=None,
                            help='Maximum allowed length for src dictionary')
    preprocess.add_argument('--max_characters_per_token', type=int, default=30,
                            help='Maximum number of characters allowed per token')
    preprocess.add_argument('--combine_train_valid', type='bool', default=False,
                            help='Combine train and valid data')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--valid_metric', type=str, default='f1',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display_iter', type=int, default=25,
                         help='Log state after every <display_iter> batches')
    general.add_argument('--sort_by_len', type='bool', default=True,
                         help='Sort batches by length for speed')
    general.add_argument('--only_test', type='bool', default=False,
                         help='Only do testing')

    # Log results Learning
    log = parser.add_argument_group('Log arguments')
    log.add_argument('--eval_mode', type='bool', default=False,
                     help='Print question id and generated answer in file')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.train_dir = os.path.join(args.data_dir, args.train_dir)
    if not args.only_test:
        if not os.path.isdir(args.train_dir):
            raise IOError('No such file: %s' % args.train_dir)

    args.valid_dir = os.path.join(args.data_dir, args.valid_dir)
    if not os.path.isdir(args.valid_dir):
        raise IOError('No such file: %s' % args.valid_dir)

    if args.vocab_file:
        args.vocab_file = os.path.join(args.data_dir, args.vocab_file)
        if not os.path.isfile(args.vocab_file):
            raise IOError('No such file: %s' % args.vocab_file)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    suffix = '_test' if args.only_test else ''
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')
    args.log_file = os.path.join(args.model_dir, args.model_name + suffix + '.txt')
    args.pred_file = os.path.join(args.model_dir, args.model_name + suffix + '.json')
    if args.pretrained:
        args.pretrained = os.path.join(args.model_dir, args.pretrained + '.mdl')

    if not args.only_test:
        if args.use_word:
            # Embeddings options
            if args.embedding_file:
                args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
                if not os.path.isfile(args.embedding_file):
                    raise IOError('No such file: %s' % args.embedding_file)

                with open(args.embedding_file) as f:
                    # if first line is of form count/dim.
                    line = f.readline().rstrip().split(' ')
                    dim = int(line[1]) if len(line) == 2 \
                        else len(line) - 1
                args.emsize = dim
            elif not args.emsize:
                raise RuntimeError('Either embedding_file or embedding_dim '
                                   'needs to be specified.')

            # Make sure tune_partial and fix_embeddings are consistent.
            if args.tune_partial > 0:
                logger.warning('WARN: fix_embeddings set to False as tune_partial > 0.')
                args.fix_embeddings = False

            # Make sure fix_embeddings and embedding_file are consistent
            if args.fix_embeddings:
                if not (args.embedding_file or args.pretrained):
                    logger.warning('WARN: fix_embeddings set to False '
                                   'as embeddings are random.')
                    args.fix_embeddings = False
        else:
            args.fix_embeddings = False
            args.tune_partial = 0

    return args


# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_from_scratch(args, train_exs, dev_exs):
    """New model, new data, new dictionary."""
    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build word dictionary')
    if args.vocab_file:
        words = set()
        with open(args.vocab_file) as f:
            for line in f:
                words.add(line.strip())
        src_dict = UnicodeCharsVocabulary(words,
                                          len(train_exs + dev_exs),
                                          args.max_characters_per_token)
    else:
        fields = ['sentence', 'question']
        src_dict = util.build_word_and_char_dict(args,
                                                 train_exs + dev_exs,
                                                 fields=fields,
                                                 dict_size=args.src_vocab_size)

    logger.info('Num words in dictionary = %d' % (len(src_dict)))

    # Initialize model
    model = QAModel(config.get_model_args(args), src_dict)

    # Load pretrained embeddings for words in dictionary
    if args.use_word and args.embedding_file:
        model.load_embeddings(args.embedding_file)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    loss = AverageMeter()
    epoch_time = Timer()

    pbar = tqdm(data_loader, total=args.num_train_batch)
    pbar.set_description("%s" % 'Epoch = %d [loss = x.xx]' % global_stats['epoch'])

    # we do not apply lr_decay in the first epoch
    if global_stats['epoch'] > 1:
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * args.lr_decay

    # Run one epoch
    for idx, ex in enumerate(pbar):
        bsz = ex['batch_size']

        if args.use_warmup_schedule and global_stats['epoch'] == 1:
            cur_lrate = args.warmup_factor * (model.updates + 1)
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = cur_lrate

        loss.update(model.update(ex), bsz)
        log_info = 'Epoch = %d loss = %.3f]' % (global_stats['epoch'], loss.avg)
        pbar.set_description("%s" % log_info)

    logger.info('train: Epoch %d | loss = %.2f | Time for epoch = %.2f (s)' %
                (global_stats['epoch'], loss.avg, epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint', global_stats['epoch'] + 1)


# ------------------------------------------------------------------------------
# Validation loops.
# ------------------------------------------------------------------------------


def validate_official(args, data_loader, model, global_stats, mode='valid'):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = Timer()
    results = []

    with torch.no_grad():
        pbar = tqdm(data_loader)
        for ex in pbar:
            output = model.predict(ex)

            for idx in range(ex['batch_size']):
                results.append(OrderedDict([
                    ('id', ex['ids'][idx]),
                    ('sid', ex['sent_id'][idx]),
                    ('qid', ex['ques_id'][idx]),
                    ('pred', output['predictions'][idx]),
                    ('gold', ex['raw_label'][idx])
                ]))
            pbar.set_description("%s" % 'Epoch = %d [validating ... ]' %
                                 global_stats['epoch'])

    results = scorer.score(results)
    prec, rec, f1 = 0.0, 0.0, 0.0
    with open(args.pred_file, 'w') as fw:
        for item in results:
            fw.write(json.dumps(item) + '\n')
            prec += item['precision']
            rec += item['recall']
            f1 += item['f1']

    total_example = len(results)
    prec, rec, f1 = prec / total_example, rec / total_example, f1 / total_example
    logger.info('Validation: precision = %.2f | recall = %.2f | f1 = %.2f |'
                ' examples = %d | %s time = %.2f (s) ' %
                (prec * 100, rec * 100, f1 * 100, total_example, mode, eval_time.time()))

    return {
        'precision': prec,
        'recall': rec,
        'f1': f1
    }


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load and process data files')

    train_exs = []
    if not args.only_test:
        train_exs = util.load_data(args.train_dir,
                                   uncase=True,
                                   dataset_name=args.dataset_name,
                                   max_examples=args.max_examples)

    dev_exs = util.load_data(args.valid_dir,
                             uncase=True,
                             dataset_name=args.dataset_name,
                             max_examples=args.max_examples)

    if not args.only_test:
        if args.combine_train_valid:
            train_exs = train_exs + dev_exs

        logger.info('Num train examples = %d' % len(train_exs))
        # IMPORTANT
        if args.filter:
            labels = [ex['one_label'] for ex in train_exs]
            pos_indices = [i for i, l in enumerate(labels) if l == 1]
            neg_indices = [i for i, l in enumerate(labels) if l == 0]
            if len(neg_indices) > len(pos_indices):
                neg_indices = neg_indices[:len(pos_indices)]
            args.num_train_examples = len(pos_indices + neg_indices)
            args.num_train_batch = int(math.ceil(args.num_train_examples / args.batch_size))
        else:
            args.num_train_examples = len(train_exs)
            args.num_train_batch = int(math.ceil(args.num_train_examples / args.batch_size))

    logger.info('Num dev examples = %d' % len(dev_exs))

    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 1
    if args.only_test:
        if args.pretrained:
            model = QAModel.load(args.pretrained)
        else:
            if not os.path.isfile(args.model_file):
                raise IOError('No such file: %s' % args.model_file)
            model = QAModel.load(args.model_file)
    else:
        if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
            # Just resume training, no modifications.
            logger.info('Found a checkpoint...')
            checkpoint_file = args.model_file + '.checkpoint'
            model, start_epoch = QAModel.load_checkpoint(checkpoint_file, args.cuda)
        else:
            # Training starts fresh. But the model state is either pretrained or
            # newly (randomly) initialized.
            if args.pretrained:
                logger.info('Using pretrained model...')
                model = QAModel.load(args.pretrained, args)
            else:
                logger.info('Training model from scratch...')
                model = init_from_scratch(args, train_exs, dev_exs)

            # Set up partial tuning of embeddings
            if args.tune_partial > 0:
                logger.info('-' * 100)
                logger.info('Counting %d most frequent question words' % args.tune_partial)
                top_words = util.top_words(
                    args, train_exs, model.src_dict
                )
                for word in top_words[:5]:
                    logger.info(word)
                logger.info('...')
                for word in top_words[-6:-1]:
                    logger.info(word)
                model.tune_embeddings([w[0] for w in top_words])

            # Set up optimizer
            model.init_optimizer()
            # log the parameter details
            logger.info('Total trainable parameters # %d' % model.network.count_parameters())
            table = model.network.layer_wise_parameters()
            logger.info('Breakdown of the trainable paramters\n%s' % table)

    # Use the GPU?
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')

    train_loader = None
    if not args.only_test:
        train_dataset = data.ReaderDataset(train_exs, model)
        train_sampler = data.SortedBatchSampler(lengths=train_dataset.lengths(),
                                                labels=train_dataset.labels(),
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                filter=args.filter)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda
        )

    dev_dataset = data.ReaderDataset(dev_exs, model)
    dev_sampler = data.SortedBatchSampler(lengths=dev_dataset.lengths(),
                                          labels=dev_dataset.labels(),
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          filter=False)

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    if not args.only_test:
        logger.info('-' * 100)
        logger.info('CONFIG:\n%s' %
                    json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # DO TEST

    if args.only_test:
        stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0, 'no_improvement': 0}
        validate_official(args, dev_loader, model, stats, mode='test')

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    else:
        logger.info('-' * 100)
        logger.info('Starting training...')
        stats = {'timer': Timer(), 'epoch': start_epoch, 'best_valid': 0, 'no_improvement': 0}

        if args.use_warmup_schedule:
            logger.info("Using warmup learning rate for the first epoch, from 0 up to %s." %
                        args.learning_rate)
        args.warmup_factor = (args.learning_rate + 0.) / args.num_train_batch

        for epoch in range(start_epoch, args.num_epochs + 1):
            stats['epoch'] = epoch

            # Train
            train(args, train_loader, model, stats)
            if args.combine_train_valid:
                model.save(args.model_file)
            else:
                result = validate_official(args, dev_loader, model, stats)
                valid_metric_perf = result[args.valid_metric]

                # Save best valid
                if valid_metric_perf > stats['best_valid']:
                    logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                                (args.valid_metric, valid_metric_perf,
                                 stats['epoch'], model.updates))
                    model.save(args.model_file)
                    stats['best_valid'] = valid_metric_perf
                    stats['no_improvement'] = 0
                else:
                    stats['no_improvement'] += 1
                    if stats['no_improvement'] >= args.early_stop:
                        break


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Question Answering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = torch.cuda.is_available()
    args.device_count = torch.cuda.device_count()
    args.parallel = args.device_count > 1

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
