# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import gc

from ProccesData import *
from Model2 import *


def generate_comp_tagged():
    print('a')

    
    model_name = 'roberta-base'  # Roberta best

    baby_tokenized_datasets,office_tokenized_datasets,unlabeled_tokenized_datasets,test_tokenized_datasets=TokenizeData(model_name)
    combinedTrain_tokenized_datasets=TokenizeAdditonalData(model_name,baby_tokenized_datasets)
    trainer=TrainCombinedModel(model_name,baby_tokenized_datasets,combinedTrain_tokenized_datasets)
    MakePredictionsOnOfficeTest(trainer, test_tokenized_datasets)

if __name__ == '__main__':
    generate_comp_tagged()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
