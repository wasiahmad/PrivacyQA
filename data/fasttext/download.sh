#!/usr/bin/env bash

FASTTEXT=./

echo "Downloading fasttext embeddings"

FILE=polisis-100d-158k-subword.txt
if [[ -f "$FILE" ]]; then
    echo "$FILE exists, skipping download"
else
    fileid="1dqrmc1in81SiqvTQPFxQdnYZdKrbpxtu"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    rm ./cookie
fi

FILE=polisis-300d-137M-subword.txt
if [[ -f "$FILE" ]]; then
    echo "$FILE exists, skipping download"
else
    fileid="1EIwu1ahCmoHkAIpnrG-fmosbu3qsCbda"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    rm ./cookie
fi
