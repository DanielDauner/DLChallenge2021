import torch
from torch.utils.data import DataLoader , Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from augmentation_utils import delete_random_patches
import matplotlib.pyplot as plt
import os

# from imgaug.augmenters import Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, \
#         Noop, Lambda, AssertLambda, AssertShape, Scale, CropAndPad, \
#         Pad, Crop, Fliplr, Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, \
#         Grayscale, GaussianBlur, AverageBlur, MedianBlur, Convolve, \
#         Sharpen, Emboss, EdgeDetect, DirectedEdgeDetect, Add, AddElementwise, \
#         AdditiveGaussianNoise, Multiply, MultiplyElementwise, Dropout, \
#         CoarseDropout, Invert, ContrastNormalization, Affine, PiecewiseAffine, \
#         ElasticTransformation

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):

        x = self.data[index].astype(np.uint8)
        x = self.transform(x)

        y = Image.fromarray(self.targets[index].astype(np.uint8))
        y = transforms.ToTensor()(y)
        return x, y

    def __len__(self):
        return len(self.data)


class AugmentationDataset(Dataset):
    def __init__(self, data, targets, background=None, aug_all=None, aug_data=None):
        
        self.data = data
        self.n_data = len(data)
        self.targets = targets
        self.background = background

        self.aug_all = aug_all
        self.aug_data = aug_data

        self.patch_proba = (0.25, 0.5)

    # def __len__(self):
    #     return 2*self.n_data
    def __len__(self):
        return self.n_data

    def __getitem__(self, index):
        
       
        if index < self.n_data:
            y = np.copy(self.targets[index].astype(np.uint8))
            x = np.copy(self.data[index].astype(np.uint8))

            # apply first augmentation to label and image
            augmentation = self.aug_all.to_deterministic()
            x, y = augmentation.augment_image(x), augmentation.augment_image(y)

            if np.random.uniform(0, 1) < self.patch_proba[0]:
                x = delete_random_patches(x)

        # else:
            
        #     new_index = index - self.n_data
        #     y = np.copy(self.targets[new_index].astype(np.uint8))
        #     x = np.copy(self.new_background(y))

        #     # apply first augmentation to label and image
        #     augmentation = self.aug_all.to_deterministic()
        #     x, y = augmentation.augment_image(x), augmentation.augment_image(y)

        #     if np.random.uniform(0, 1) < self.patch_proba[1]:
        #         x = delete_random_patches(x)

        # apply second augmentation only to image
        x = self.aug_data.augment_image(x)

        x, y = transforms.ToTensor()(x.copy()), transforms.ToTensor()(y.copy()) 

        return x, y

    def new_background(self, label):
    
        new_image = np.copy(self.background[np.random.randint(0,10000)])
        greyscale_image = np.mean(label, axis=2)
        mask = greyscale_image > 0
        new_image[mask,:] = label[mask,:]

        return new_image


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




def get_dataloaders(batch_size, validation_fraction = 0.1, train_data_path = 'data/train_noisy.npy',train_labels_path = 'data/train_clean.npy',test_data_path = 'data/test_noisy_100.npy',background_data_path = 'data/train_background.npy'):
    """
    :param batch_size:
    :param validation_fraction: usually 10 percent of the training set. In the final run you want to submit you can set it to ß,since more date will increase the training performance.
    :param train_data_path: See Kaggle challenge for these files
    :param train_labels_path:
    :param test_data_path:
    :return: train_dataloader,val_dataloader,test_dataloader.  The retuned labels of the test dataloader are zero tensors.
    """
    
    test_data = np.load(test_data_path)


    full_train_set = DatasetLoad("/home/daniel/dataset", max_size=500_000)

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