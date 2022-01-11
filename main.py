# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from ProccesData import *
from Model import *


def print_hi(name):

    model_name = 'roberta-base'  # Roberta best
    baby_tokenized_datasets,office_tokenized_datasets,unlabeled_tokenized_datasets,test_tokenized_datasets=TokenizeData(model_name)
    trainer=TrainModel(model_name,baby_tokenized_datasets)
    BabyPredictionsDev(trainer, baby_tokenized_datasets)
    OfficePredictionsDev(trainer, office_tokenized_datasets)
    #AddUnlabledData(trainer, unlabeled_tokenized_datasets) # takes around 50 mins
    #combinedTrain_tokenized_datasets=TokenizeAdditonalData(model_name)
    #TrainCombinedModel(model_name,baby_tokenized_datasets,combinedTrain_tokenized_datasets)
    TokenizeAdditonalData(model_name,baby_tokenized_datasets)
    MakePredictionsOnOfficeTest(trainer, test_tokenized_datasets)

if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
