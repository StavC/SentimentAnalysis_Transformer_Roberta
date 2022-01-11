import numpy as np
from pathlib import Path
import torch
torch.manual_seed(0)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, load_metric
def TokenizeData(model_name):

    baby_train_path = 'baby/babyTrain.csv'
    baby_dev_path = 'baby/babyDev.csv'
    office_dev_path = 'office/office_dev.csv'
    office_test_path = 'office/office_test.csv'

    baby_unlabeled_path = 'baby/babyUnlabeled.csv'
    office_unlabeled_path = 'office/office_unlabeled.csv'

    data_files = {
        'train': str(baby_train_path),
        'test': str(baby_dev_path),
    }
    baby_datasets = load_dataset("csv", data_files=data_files)

    baby_datasets = baby_datasets.shuffle(seed=5)

    office_data_files = {
        'dev': str(office_dev_path)
    }
    office_datasets = load_dataset("csv", data_files=office_data_files)
    office_datasets = office_datasets.shuffle(seed=15)

    unlabeled_data_files = {
        'baby': str(baby_unlabeled_path),
        'office': str(office_unlabeled_path)
    }
    unlabeled_datasets = load_dataset("csv", data_files=unlabeled_data_files)


   ########################################
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    baby_tokenized_datasets = baby_datasets.map(tokenizer, input_columns='review',
                                                fn_kwargs={"max_length": 200, "truncation": True,
                                                           "padding": "max_length"})
    baby_tokenized_datasets.set_format('torch')

    # office
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    office_tokenized_datasets = office_datasets.map(tokenizer, input_columns='review',
                                                    fn_kwargs={"max_length": 200, "truncation": True,
                                                               "padding": "max_length"})
    office_tokenized_datasets.set_format('torch')

    # unlabeled
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    unlabeled_tokenized_datasets = unlabeled_datasets.map(tokenizer, input_columns='review',
                                                          fn_kwargs={"max_length": 200, "truncation": True,
                                                                     "padding": "max_length"})
    unlabeled_tokenized_datasets.set_format('torch')
    ########################################
    # baby add label
    for split in baby_tokenized_datasets:
        baby_tokenized_datasets[split] = baby_tokenized_datasets[split].add_column('label',
                                                                                   baby_datasets[split]['label'])
    print(baby_tokenized_datasets)

    # office Add label
    for split in office_tokenized_datasets:
        office_tokenized_datasets[split] = office_tokenized_datasets[split].add_column('label',
                                                                                       office_datasets[split]['label'])
    print(office_tokenized_datasets)

    # unlabeled Split
    for split in unlabeled_tokenized_datasets:
        print(unlabeled_tokenized_datasets)

    return baby_tokenized_datasets,office_tokenized_datasets,unlabeled_tokenized_datasets