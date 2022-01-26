import os
import time
import numpy as np
import torch
import data_handling
import argument_handling
import models
import functools
from torchvision.utils import save_image
import torchvision.transforms as transforms
import sys
import matplotlib.pyplot as plt
from util import mse_loss

torch.backends.cudnn.deterministic = True

def main():

    args = argument_handling.parse_args()
    train_dataloader, val_dataloader, test_dataloader = \
        data_handling.get_dataloaders(args.batch_size,args.validation_fraction,
                                      args.train_data_path,args.train_labels_path,args.test_data_path)

    
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = models.BetaUNet2Large().to(device)
    model.load_state_dict(torch.load(args.model_path))


    print(f"Sucesfully loaded model from {args.model_path}") 


    if args.predicted_test_label_save_dir:
        save_predicted_test_labels(args.predicted_test_label_save_dir, test_dataloader, model, device)

    print("Done!")



def validate(val_dataloader, model, loss_fn, device):
    num_batches = len(val_dataloader)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, Y in val_dataloader:
            X, Y = X.to(device), Y.to(device)
            Y = torch.flatten(Y, 1)
            Y_flat = torch.flatten(Y, 1)
            pred_X,_,__ = model(X)
            pred_flat = torch.flatten(pred_X, 1)
            val_loss += loss_fn(torch.round(pred_flat*255), Y_flat*255).item()
    val_loss /= num_batches
    print(f"Validation loss: {val_loss:>7f} ")
    print("-" * 50)



def save_predicted_test_labels(predicted_test_label_save_dir,test_dataloader,model,device):
    model.eval()
    predictions=[]
    path = predicted_test_label_save_dir+"/kaggle_prediction_"+time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    os.mkdir(path)
    with torch.no_grad():
        for i, (X, Y) in enumerate(test_dataloader):
            X = X.to(device)
            pred_X,_,__ = model(X)

            flat_pred_X = pred_X.cpu().numpy().flatten()
            predictions = np.concatenate((predictions, flat_pred_X))
            pic_num = int(i*test_dataloader.batch_size)
            for j, (train_img,test_img) in enumerate(zip(X,pred_X)):
                save_image(train_img,f"{path}/{j+pic_num}_train.png")
                save_image(test_img,f"{path}/{j+pic_num}_predicted.png")

    predictions *=255
    
    predictions = np.expand_dims(predictions, 1)
    indices = np.expand_dims(np.arange(len(predictions)), 1)
    csv_data = np.concatenate([indices, predictions], axis=1)
    csv_file = path+".csv"
    np.savetxt(csv_file, csv_data, delimiter=",", header='Id,Value', fmt='%d,%f')

if __name__=="__main__":
    main()

