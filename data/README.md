### PrivacyQA

If the [download.sh](https://github.com/wasiahmad/PrivacyQA/blob/master/data/privacyQA/download.sh) doesn't work properly, please download the dataset from the authors' provided [link](https://drive.google.com/file/d/1tUdsVqp9-8Tr8w2prQ5MlUhJ8zUa6ZdA/view).

### Fasttext embeddings

We train word embeddings using [fastText](https://fasttext.cc/) based on a corpus of 130K privacy policies (137M words) 
collected from apps on the Google Play Store. Download the fasttext embeddings by running the 
[download.sh](https://github.com/wasiahmad/PrivacyQA/tree/master/data/fasttext) file.


### BERT-pretrained weights

Download all the required files for five different sizes [tiny, mini, small, medium, base] of BERT by running [download.sh](https://github.com/wasiahmad/PrivacyQA/blob/master/data/bert/original/download.sh) script. Read this [paper](https://arxiv.org/pdf/1908.08962.pdf) to learn about the different sizes of BERT.

#### Fine-tuning using MLM

Similarly, to adapt BERT to the security and privacy domain, we fine-tune BERT-base using [masked language modeling](https://arxiv.org/abs/1810.04805)
based on the 130k privacy policies. To use the pre-trained BERT fine-tuned on privacy policies, download the required files by running the 
[download.sh](https://github.com/wasiahmad/PrivacyQA/blob/master/data/bert/pretrained/download.sh) script.


### Acknowledgement

We thank the authors of [Harkous et al., 2018](https://arxiv.org/abs/1802.02561) for sharing the 130k privacy policies 
with us.

