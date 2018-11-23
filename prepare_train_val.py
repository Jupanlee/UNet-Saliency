import glob
from prepare_data import (train_path, valid_path)

def get_split(fold):

    train_file_names = []
    val_file_names = []

    val_file_names = glob.glob(valid_path+'/image/*.png')
    train_file_names = glob.glob(train_path+'/image/*.jpg')

    return train_file_names, val_file_names
