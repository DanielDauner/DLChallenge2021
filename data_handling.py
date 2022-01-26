import torch
from torch.utils.data import DataLoader , Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import os


class TestDataset(Dataset):
    def __init__(self, data, targets):
        
        self.data = data
        self.n_data = len(data)
        self.targets = targets

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):
         
        y = self.targets[index].astype(np.uint8)
        x = self.data[index].astype(np.uint8)

        x, y = transforms.ToTensor()(x.copy()), transforms.ToTensor()(y.copy()) 

        return x, y

class DatasetLoad(Dataset):
    def __init__(self, datadir, max_size=None):
        
        self.datadir = datadir
        self.n = int(len(os.listdir(datadir))) if max_size == None else max_size

    def __len__(self):
        return self.n

    def __getitem__(self, index):

        sample = np.load(os.path.join(self.datadir, f"sample_{index}.npy")).astype(np.uint8)
        x, y = transforms.ToTensor()(sample[0]), transforms.ToTensor()(sample[1]) 

        return x, y

def get_dataloaders(batch_size, validation_fraction = 0.05, train_data_path = 'data/train_noisy.npy',train_labels_path = 'data/train_clean.npy',test_data_path = 'data/test_noisy_100.npy',background_data_path = 'data/train_background.npy'):
    """
    :param batch_size:
    :param validation_fraction: usually 10 percent of the training set. In the final run you want to submit you can set it to ß,since more date will increase the training performance.
    :param train_data_path: See Kaggle challenge for these files
    :param train_labels_path:
    :param test_data_path:
    :return: train_dataloader,val_dataloader,test_dataloader.  The retuned labels of the test dataloader are zero tensors.
    """
    
    test_data = np.load(test_data_path)

    full_train_set = DatasetLoad(train_data_path, max_size=2_000_000)

    train_size = int(len(full_train_set)*(1-validation_fraction))
    val_size = len(full_train_set)-train_size
    lengths = [train_size, val_size]
    train_set, val_set = torch.utils.data.dataset.random_split(full_train_set, lengths,
                                                            torch.Generator().manual_seed(42))
    
    test_set = TestDataset(test_data, np.zeros_like(test_data))
    
    num_parallel_loading_threads=4
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=num_parallel_loading_threads, pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True,num_workers=num_parallel_loading_threads, pin_memory=True)
    
    
    test_dataloader = DataLoader(test_set, batch_size=batch_size)


    return train_dataloader,val_dataloader,test_dataloader