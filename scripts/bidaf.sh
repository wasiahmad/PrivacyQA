#!/usr/bin/env bash

SRC_DIR=..
DATA_DIR=${SRC_DIR}/data
EMBED_DIR=${SRC_DIR}/data/fasttext
MODEL_DIR=${SRC_DIR}/tmp

if [[ ! -d $DATA_DIR ]]; then
    echo "${DATA_DIR} does not exist"
    exit 1
fi

if [[ ! -d $EMBED_DIR ]]; then
    echo "${EMBED_DIR} does not exist"
    exit 1
fi

if [[ ! -d $MODEL_DIR ]]; then
    echo "${MODEL_DIR} does not exist, creating the directory"
    mkdir $MODEL_DIR
fi

DATASET=privacyQA
EMB_FILE=polisis-300d-137M-subword.txt

function train () {

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/bidaf/main.py \
--data_workers 5 \
--combine_train_valid False \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/${DATASET}/ \
--train_dir train/ \
--valid_dir valid/ \
--filter False \
--vocab_file vocab.txt \
--embed_dir $EMBED_DIR \
--embedding_file $EMB_FILE \
--tune_partial 0 \
--fix_embeddings True \
--restrict_vocab False \
--batch_size 32 \
--test_batch_size 32 \
--max_sent_len 200 \
--num_epochs 20 \
--rnn_type LSTM \
--nhid 300 \
--nlayers 1 \
--dropout_emb 0.2 \
--dropout_rnn 0.2 \
--dropout 0.2 \
--pos_weight 1.0 \
--early_stop 3 \
--optimizer adam \
--learning_rate 0.001 \
--grad_clipping 1.0 \
--lr_decay 0.95 \
--checkpoint True \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--valid_metric f1

}

function test () {

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/bidaf/main.py \
--only_test True \
--data_workers 5 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/${DATASET}/ \
--valid_dir test/ \
--embed_dir $EMBED_DIR \
--embedding_file $EMB_FILE \
--test_batch_size 32 \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME

}

train $1 $2
test $1 $2
