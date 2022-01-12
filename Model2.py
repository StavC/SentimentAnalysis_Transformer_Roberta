import gc
from pathlib import Path
import torch
torch.manual_seed(0)
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from Metrics import metric_fn
import pandas as pd
import GPUtil
from transformers import RobertaConfig, RobertaModel

def TrainModel(model_name,baby_tokenized_datasets):
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    torch.cuda.empty_cache()
    print(GPUtil.showUtilization())

    model_seq_classification1 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    print(GPUtil.showUtilization())

    OUT_PATH = Path("trainDir")

    args = TrainingArguments(output_dir='OUT_PATH', overwrite_output_dir=True, per_device_train_batch_size=32,
                             per_device_eval_batch_size=32, save_strategy='epoch',
                             metric_for_best_model='eval_accuracy', load_best_model_at_end=True, greater_is_better=True,
                             evaluation_strategy='epoch', do_train=True,
                             num_train_epochs=5)
    print(GPUtil.showUtilization())

    trainer = Trainer(
        model=model_seq_classification1,
        args=args,
        train_dataset=baby_tokenized_datasets['train'],
        eval_dataset=baby_tokenized_datasets['test'],
        compute_metrics=metric_fn
    )
    trainer.train()
    print('finished training vanila model')
    return trainer

def BabyPredictionsDev(trainer,baby_tokenized_datasets):
    baby_pred = trainer.predict(test_dataset=baby_tokenized_datasets['test'])[0]

    preds = baby_pred.argmax(axis=1)

    true_labels = baby_tokenized_datasets['test']['label']
    acc = accuracy_score(true_labels, preds)
    print('Baby Predictions on DEV')
    print(true_labels, preds, acc)

def OfficePredictionsDev(trainer,office_tokenized_datasets):
    office_pred = trainer.predict(test_dataset=office_tokenized_datasets['dev'])[0]

    preds = office_pred.argmax(axis=1)
    true_labels = office_tokenized_datasets['dev']['label']
    acc = accuracy_score(true_labels, preds)
    print('Baby Office on DEV')

    print(true_labels, preds, acc)

def TrainCombinedModel(model_name,baby_tokenized_datasets,combinedTrain_tokenized_datasets):
    OUT_PATH = Path("trainDir")
    print('Starting to train Combined Model')

    model_seq_classification2 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(output_dir=OUT_PATH, overwrite_output_dir=True, per_device_train_batch_size=16,
                             per_device_eval_batch_size=16, save_strategy='epoch',
                             metric_for_best_model='eval_accuracy', load_best_model_at_end=True, greater_is_better=True,
                             evaluation_strategy='epoch', do_train=True,
                             num_train_epochs=5, report_to='none')
    trainer = Trainer(
        model=model_seq_classification2,
        args=args,
        train_dataset=combinedTrain_tokenized_datasets['train'],
        eval_dataset=baby_tokenized_datasets['test'],
        compute_metrics=metric_fn
    )
    trainer.train()
    print('Finished training Combined Model')
    return trainer

def MakePredictionsOnOfficeTest(trainer,test_tokenized_datasets):
    test_pred = trainer.predict(test_dataset=test_tokenized_datasets['test'])[0]
    preds = test_pred.argmax(axis=1)

    office_test_labeled = pd.read_csv('office/office_test.csv')
    office_test_labeled.columns.values[0] = ''
    office_test_labeled['label'] = preds
    office_test_labeled['label'] = [True if row is 1 else False for row in office_test_labeled['label']]
    office_test_labeled.to_csv('office_test_labeled.csv', index=False)
    print('finished writing office_test_labeled')