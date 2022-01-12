import numpy as np
from pathlib import Path
import torch
torch.manual_seed(0)
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from Metrics import metric_fn
from transformers import AutoTokenizer
#from transformers import RobertaTokenizer
import gc
import GPUtil
from datasets import load_dataset, load_metric
import pandas as pd
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

    office_data = {
        'test': str(office_test_path)
    }
    office_datasets_test = load_dataset("csv", data_files=office_data)
    print(GPUtil.showUtilization())
   ########################################
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    baby_tokenized_datasets = baby_datasets.map(tokenizer, input_columns='review',
                                                fn_kwargs={"max_length": 350, "truncation": True,
                                                           "padding": "max_length"})
    baby_tokenized_datasets.set_format('torch')

    # office
    office_tokenized_datasets = office_datasets.map(tokenizer, input_columns='review',
                                                    fn_kwargs={"max_length": 350, "truncation": True,
                                                               "padding": "max_length"})
    office_tokenized_datasets.set_format('torch')

     # unlabeled
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    unlabeled_tokenized_datasets = unlabeled_datasets.map(tokenizer, input_columns='review',
                                                          fn_kwargs={"max_length": 350, "truncation": True,
                                                                     "padding": "max_length"})
    unlabeled_tokenized_datasets.set_format('torch')
    # test
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_tokenized_datasets = office_datasets_test.map(tokenizer, input_columns='review',
                                                       fn_kwargs={"max_length": 350, "truncation": True,
                                                                  "padding": "max_length"})
    test_tokenized_datasets.set_format('torch')
    print(GPUtil.showUtilization())
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

    #unlabeled Split
    for split in unlabeled_tokenized_datasets:
        print(unlabeled_tokenized_datasets)

    for split in office_datasets_test:
        print(test_tokenized_datasets)
    print('finished Tokenizing')
    print(GPUtil.showUtilization())

    return baby_tokenized_datasets,office_tokenized_datasets,unlabeled_tokenized_datasets,test_tokenized_datasets


def AddUnlabledData(trainer,unlabeled_tokenized_datasets):
    print('Working on predicting the UnlabledData')
    office_pred = trainer.predict(test_dataset=unlabeled_tokenized_datasets['baby'])[0]
    preds = office_pred.argmax(axis=1)
    print(preds)

    babyUnlabeled = pd.read_csv('baby/babyUnlabeled.csv')
    babyUnlabeled['preds'] = preds
    babyUnlabeled.to_csv('baby_unlabeled_labeled.csv', index=False)

    office_pred = trainer.predict(test_dataset=unlabeled_tokenized_datasets['office'])[0]
    preds = office_pred.argmax(axis=1)
    print(preds)

    officeUnlabeled = pd.read_csv('office/office_unlabeled.csv')
    officeUnlabeled['preds'] = preds
    officeUnlabeled.to_csv('office_unlabeled_labeled.csv', index=False)
    print('finished predicting the unlabledData')

def TokenizeAdditonalData(model_name,baby_tokenized_datasets):
    baby_df = pd.read_csv('baby/babyTrain.csv')

    baby_unlabeled_labeled = pd.read_csv('baby_unlabeled_labeled.csv')
    office_unlabeled_labeled = pd.read_csv('office_unlabeled_labeled.csv')

    # random sampling
    baby_sample_size = 1000
    office_sample_size = 1000

    baby_sampled = baby_unlabeled_labeled.sample(n=baby_sample_size, random_state=233)
    office_sampled = office_unlabeled_labeled.sample(n=office_sample_size, random_state=23)

    baby_sampled = baby_sampled.rename({'review': 'review', 'preds': 'label'}, axis=1)
    office_sampled = office_sampled.rename({'review': 'review', 'preds': 'label'}, axis=1)

    # concat to one df
    combinedTrain = pd.concat([baby_df, baby_sampled, office_sampled])
    combinedTrain = combinedTrain.drop(['Unnamed: 0'], axis=1)
    print(combinedTrain.head())

    #combinedTrain.to_csv('combinedTrain.csv', index=False)
    combinedTrain = pd.read_csv('combinedTrain.csv')

    combinedTrain_path = 'combinedTrain.csv'
    combinedTrain_files = {
        'train': combinedTrain_path
    }
    combinedTrain_datasets = load_dataset("csv", data_files=combinedTrain_files)

    # combined token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    combinedTrain_tokenized_datasets = combinedTrain_datasets.map(tokenizer, input_columns='review',
                                                                  fn_kwargs={"max_length": 350, "truncation": True,
                                                                             "padding": "max_length"})
    combinedTrain_tokenized_datasets.set_format('torch')

    for split in combinedTrain_tokenized_datasets:
        combinedTrain_tokenized_datasets[split] = combinedTrain_tokenized_datasets[split].add_column('label',
                                                                                                     combinedTrain_datasets[
                                                                                                         split][
                                                                                                         'label'])
    print(combinedTrain_tokenized_datasets)
    return combinedTrain_tokenized_datasets