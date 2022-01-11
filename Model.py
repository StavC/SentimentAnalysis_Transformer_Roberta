from pathlib import Path

from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from Metrics import metric_fn

def TrainModel(model_name,baby_tokenized_datasets):

    model_seq_classification = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    OUT_PATH = Path("trainDir")

    args = TrainingArguments(output_dir=OUT_PATH, overwrite_output_dir=True, per_device_train_batch_size=32,
                             per_device_eval_batch_size=64, save_strategy='epoch',
                             metric_for_best_model='eval_accuracy', load_best_model_at_end=True, greater_is_better=True,
                             evaluation_strategy='epoch', do_train=True,
                             num_train_epochs=5, report_to='none')
    trainer = Trainer(
        model=model_seq_classification,
        args=args,
        train_dataset=baby_tokenized_datasets['train'],
        eval_dataset=baby_tokenized_datasets['test'],
        compute_metrics=metric_fn
    )
    trainer.train()
    return trainer


