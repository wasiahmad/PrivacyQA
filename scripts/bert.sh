#!/usr/bin/env bash

SRC_DIR=..
DATA_DIR=${SRC_DIR}/data
MODEL_DIR=${SRC_DIR}/tmp

if [[ ! -d $DATA_DIR ]]; then
    echo "${DATA_DIR} does not exist"
    exit 1
fi

if [[ ! -d $MODEL_DIR ]]; then
    echo "${MODEL_DIR} does not exist, creating the directory"
    mkdir $MODEL_DIR
fi

DATASET=privacyQA

function train () {

RGPU=$1
MODEL_NAME=$2
BERT_SIZE=base

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/bert/main.py \
--fp16 True \
--combine_train_valid True \
--data_workers 5 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/${DATASET}/ \
--train_dir train/ \
--valid_dir valid/ \
--filter False \
--max_sent_len 200 \
--bert_dir ${DATA_DIR}/bert/original/ \
--bert_model bert_${BERT_SIZE}_uncased \
--batch_size 32 \
--test_batch_size 32 \
--num_epochs 3 \
--pos_weight 1.0 \
--early_stop 5 \
--gradient_accumulation_steps 1 \
--warmup_steps 0 \
--learning_rate 2e-5 \
--grad_clipping 1.0 \
--checkpoint True \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--valid_metric f1

}

function test () {

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/bert/main.py \
--only_test True \
--data_workers 5 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/${DATASET}/ \
--valid_dir test/ \
--bert_dir ${DATA_DIR}/bert/original/ \
--test_batch_size 32 \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME

}

train $1 $2
test $1 $2
