import torch
from torch.utils.data import DataLoader , Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from augmentation_utils import delete_random_patches
import matplotlib.pyplot as plt
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

        x=self.data[index].astype(np.uint8)
        x= self.transform(x)

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

    def __len__(self):
        return 2*self.n_data

    def __getitem__(self, index):
        
       
        if index < self.n_data:
            y = self.targets[index].astype(np.uint8)
            x = self.data[index].astype(np.uint8)
        else:
            new_index = index - self.n_data
            y = self.targets[new_index].astype(np.uint8)
            x = self.new_background(y)


        # apply first augmentation to label and image
        augmentation = self.aug_all.to_deterministic()
        x, y = augmentation.augment_image(x), augmentation.augment_image(y)

        # apply second augmentation only to image
        x, y = self.aug_data.augment_image(x), self.aug_data.augment_image(y)

        x, y = transforms.ToTensor()(x), transforms.ToTensor()(y) 


        return x, y

    def new_background(self, label):
    
        new_image = self.background[np.random.randint(0,10000)]
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

        x, y = transforms.ToTensor()(x), transforms.ToTensor()(y) 

        return x, y




def get_dataloaders(batch_size,validation_fraction = 0.9, train_data_path = 'data/train_noisy.npy',train_labels_path = 'data/train_clean.npy',test_data_path = 'data/test_noisy_100.npy',background_data_path = 'data/train_background.npy'):
    """
    :param batch_size:
    :param validation_fraction: usually 10 percent of the training set. In the final run you want to submit you can set it to ß,since more date will increase the training performance.
    :param train_data_path: See Kaggle challenge for these files
    :param train_labels_path:
    :param test_data_path:
    :return: train_dataloader,val_dataloader,test_dataloader.  The retuned labels of the test dataloader are zero tensors.
    """
    test_data = np.load(test_data_path)
    train_data = np.load(train_data_path)
    train_labels = np.load(train_labels_path)
    background_data = np.load(train_labels_path)


    # transform_train = transforms.Compose([transforms.ToTensor()])
    augmentation_all = iaa.Sequential([
                    iaa.Sometimes(0.5, iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})),
                    iaa.Sometimes(0.5, iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})),
                    iaa.Sometimes(0.5, iaa.Affine(rotate=(-90, 90))),
                    iaa.Sometimes(0.5, iaa.Affine(shear=(-25, 25))),
                    iaa.Sometimes(0.5, iaa.Fliplr()),
                    iaa.Sometimes(0.5, iaa.Flipud()),
                    iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.1, 0.1))),
                    ], random_order=True)
    

    augmentation_data = iaa.Sequential([
                    iaa.Sometimes(0.5,iaa.OneOf([
                        iaa.GaussianBlur((0, 1.5)),
                        iaa.AverageBlur(k=(2, 3)),
                    ])),
                    iaa.Sometimes(0.5,iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.03, 0.15), size_percent=(0.02, 0.03),
                            per_channel=0.2
                        ),
                    ]),),
                    # iaa.Sometimes(0.25,iaa.Superpixels(
                    #         p_replace=(0, 0.5),
                    #         n_segments=(20, 100)
                    #     )),
                    iaa.Sometimes(0.5,iaa.Invert(0.05, per_channel=True)),
                    iaa.Sometimes(0.5,iaa.Add((-20, 20), per_channel=0.5)),
                    iaa.Sometimes(0.5,iaa.Multiply((0.5, 1.5), per_channel=0.5)),
                    iaa.Sometimes(0.5,iaa.LinearContrast((0.5, 2.0), per_channel=0.5)),
                    iaa.Sometimes(0.5,iaa.Grayscale(alpha=(0.0, 1.0))),
                    
                    ], random_order=True)
    





    full_train_set = AugmentationDataset(train_data, train_labels, background=background_data, aug_all=augmentation_all, aug_data=augmentation_data)

    train_size = int(len(full_train_set)*(1-validation_fraction))
    val_size = len(full_train_set)-train_size
    lengths = [train_size, val_size]
    train_set, val_set = torch.utils.data.dataset.random_split(full_train_set, lengths,
                                                               torch.Generator().manual_seed(42))

    test_set = TestDataset(test_data, np.zeros_like(test_data))
    
    
    num_parallel_loading_threads=4
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=num_parallel_loading_threads)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True,num_workers=num_parallel_loading_threads)
    test_dataloader = DataLoader(test_set, batch_size=batch_size)

    return train_dataloader,val_dataloader,test_dataloader