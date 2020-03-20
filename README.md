# PrivacyQA
Unofficial model implementations for the [PrivacyQA](https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP) benchmark.

### Requirements

python 3.6, [pytorch 1.3](https://pytorch.org/get-started/previous-versions/#commands-for-versions--100), [spaCy](https://spacy.io/usage), [tqdm](https://pypi.org/project/tqdm/), [prettytable](https://pypi.org/project/PrettyTable/), [boto3](https://github.com/boto/boto3)

### Setup

Before training/testing models on PrivacyQA, we need to run the followings.

- Download [bert weight files](https://github.com/wasiahmad/PrivacyQA/tree/master/data/bert) and [fasttext embeddings](https://github.com/wasiahmad/PrivacyQA/tree/master/data/fasttext).
- Download and preprocess the PrivacyQA [dataset](https://github.com/wasiahmad/PrivacyQA/blob/master/data/privacyQA/download.sh).

### Training/Testing

We have two QA model (BERT and BiDAF) implementation in this repo. 

In the BERT based QA model, we use the pretrained weights from one of the five BERT models - ['bert_tiny_uncased', 'bert_mini_uncased', 'bert_small_uncased', 'bert_medium_uncased', 'bert_base_uncased']. To use one of them, set the [bert_model](https://github.com/wasiahmad/PrivacyQA/blob/master/scripts/bert.sh#L34) flag accordingly.


To train and test them, go to the [scripts](https://github.com/wasiahmad/PrivacyQA/tree/master/scripts) folder and run the bash files (bidaf.sh and bert.sh).

```
$ cd  scripts
$ bash bert.sh gpu_id model_file
```

Here, `model_file` is just a string that will be used to name the model file where the model states will be saved.

Once training/testing is finished, inside the `tmp/` directory, 6 files will appear.
- model_file.json [[Contains the predictions and gold references along with an unique id.]]()
- model_file.txt [[Log file for training.]]()
- model_file.mdl [[Best model file.]]()
- model_file.mdl.checkpoint [[A model checkpoint, in case if we need to restart the experiment.]]()
- model_file_test.json [[Similar to model_file.json, but for testing.]]()
- model_file_test.txt [[Log file for testing.]]()

**[Structure of the JSON files]** Each line in a json file looks like follows.

```
{
    "id": "brilliant.13", 
    "predictions": [31, 80], 
    "gold": {"0": [31]}, 
    "precision": 0.5, 
    "recall": 1.0, 
    "f1": 0.6666666666666666
}
```

Here, `id` is of the format `[policy_name].[question_id]`.

### Important Hyper-parameters

- `filter` [[Filter training data to balance positive/negative examples]](https://github.com/wasiahmad/PrivacyQA/blob/master/nqa/inputters/dataset.py#L63)
- `pos_weight` [[Weight of the positive examples in the loss function]](https://github.com/wasiahmad/PrivacyQA/blob/master/bert/model.py#L59)
- `combine_train_valid` [[Combine train and valid data]]()
  - With this flag enabled, no validation is performed. At every epoch, the model is saved.

### Evaluation

According to the original work, precision and recall are implemented by measuring the overlap between predicted sentences and sets of gold-reference sentences. The average of the maximum F1 from each n−1 subset, in relation to the heldout reference is reported. Checkout the evaluation [script](https://github.com/wasiahmad/PrivacyQA/blob/master/nqa/eval/scorer.py).

**[Note.]** If predicted sentence and gold-reference sentence lists are empty (which means the question is unanswerable), we set the precision, recall, and f1 to 1.0.

### Results

After running the bash script to train/test model, we can get the final result in `model_file_test.txt` file. For example, for the BERT-based QA model, we get the following results.

```
[ precision = 49.95 | recall = 40.05 | f1 = 39.04 | examples = 400  ]
```

And the above mentioned F1 score the performance reported in the [paper](https://arxiv.org/abs/1911.00841) (check Table 6, row with model **BERT**).

For the BiDAF model, we get the following results.
```
[ precision = 33.71 | recall = 26.40 | f1 = 27.22 | examples = 400  ]
```

#### Human Performance

To compute the human performance as reported in the paper, run:

```
$ python human_performance.py
```

We get the following performance as reported in the [paper](https://arxiv.org/abs/1911.00841) (check Table 6, row with model **Human**).

```
[ precision = 68.81 | recall = 69.04 | f1 = 68.92 | examples = 400 ]
```

#### Overall Result

| Attribute                 | Precision | Recall | F1 |
| :--- | ---: | ---: | ---: |
| Human Performance         | 68.81 | 69.04 | 68.92 |
| [BiDAF](https://arxiv.org/abs/1611.01603)                     | 33.71 | 26.40 | 27.22 |
| [BERT-tiny](https://arxiv.org/abs/1908.08962)                 | 24.75 | 24.75 | 24.75 |
| [BERT-mini](https://arxiv.org/abs/1908.08962)                 | 29.21 | 26.73 | 26.71 |
| [BERT-small](https://arxiv.org/abs/1908.08962)                | 40.68 | 32.68 | 33.53 |
| [BERT-medium](https://arxiv.org/abs/1908.08962)               | 43.55 | 34.61 | 34.57 |
| [BERT-base](https://arxiv.org/abs/1908.08962)                 | 44.95 | 37.24 | 36.50 |
| BERT-base-LM-fine-tuned   | 49.95 | 40.05 | 39.04 |


### Running experiments on CPU/GPU/Multi-GPU

- If `gpu_id` is set to -1, CPU will be used.
- If `gpu_id` is set to one specific number, only one GPU will be used.
- If `gpu_id` is set to multiple numbers (e.g., 0,1,2), then parallel computing will be used.

### Acknowledgement

We borrowed and modified code from [DrQA](https://github.com/facebookresearch/DrQA), [OpenNMT](https://github.com/OpenNMT/OpenNMT-py), and [Transformers](https://github.com/huggingface/transformers). I would like to expresse my gratitdue for authors of these repositeries.

