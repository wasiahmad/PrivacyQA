#!/usr/bin/env bash

OUTDIR=bert_base_uncased
if [[ ! -d $OUTDIR ]]; then
    FILE=config.json
    if [[ ! -f "$FILE" ]]; then
        wget -O ${OUTDIR}/${FILE} https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
    fi
    FILE=vocab.txt
    if [[ ! -f "$FILE" ]]; then
        wget -O ${OUTDIR}/${FILE} https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
    fi
    FILE=pytorch_model.bin
    if [[ ! -f "$FILE" ]]; then
        fileid="16TujtANI5HSAvMt90vrMLDu5QPLk-pr3"
        curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
        curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${OUTDIR}/${FILE}
        rm ./cookie
    fi
fi


