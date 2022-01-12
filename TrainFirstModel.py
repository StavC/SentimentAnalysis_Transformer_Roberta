# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import gc

from ProccesData import *
from Model2 import *
import GPUtil


def set_seed(seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def trainFirstModel():
    set_seed()
    model_name = 'roberta-base'  # Roberta best

    baby_tokenized_datasets,office_tokenized_datasets,unlabeled_tokenized_datasets,test_tokenized_datasets=TokenizeData(model_name)
    trainer=TrainModel(model_name,baby_tokenized_datasets)
    BabyPredictionsDev(trainer, baby_tokenized_datasets)
    OfficePredictionsDev(trainer, office_tokenized_datasets)
    AddUnlabledData(trainer, unlabeled_tokenized_datasets)

if __name__ == '__main__':
    trainFirstModel()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
