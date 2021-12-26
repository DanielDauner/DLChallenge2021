import os
import time
import numpy as np
import torch
import data_handling
import argument_handling
import models
import functools
from torchvision.utils import save_image
import sys
import matplotlib.pyplot as plt
from util import mse_loss, VisualizeTraining


torch.backends.cudnn.deterministic = True

def main():

    args = argument_handling.parse_args()
    train_dataloader, val_dataloader, test_dataloader = \
        data_handling.get_dataloaders(args.batch_size,args.validation_set_size,
                                      args.train_data_path,args.train_labels_path,args.test_data_path)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    learning_rate = args.learning_rate
    epochs = args.epochs
    bootstrap = args.bootstrap

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")

    #model = models.SimpleAutoencoder().to(device)
    model = models.UNet().to(device)
    
    # TODO: enable pretraining
    #if args.pretrain_path:

    print(model)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), learning_rate,momentum=0.9)

    Visualize = VisualizeTraining("training.png", epochs)
   

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        print("-" * 50)
        start = time.time()
        train_loss = train_one_epoch(train_dataloader, model, mse_loss, optimizer, device)
        
        print("Time per Epoch: ", time.time()-start)

        val_loss = validate(val_dataloader, model, mse_loss, device)
        Visualize.update(train_loss, val_loss, epoch)

        sys.stdout.flush()
        sys.stderr.flush()
    
    if args.model_path:
        torch.save(model.state_dict(), args.model_path)
        print(f"Sucesfully saved model to {args.model_path}")    

    print("Done!")



def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    epoch_loss= 0.0
    for batch_number, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)

        Y_flat = torch.flatten(Y,1)
        pred = model(X)
        pred_flat = torch.flatten(pred, 1)
        loss = loss_fn(pred_flat*255, Y_flat*255) # to be consistent with the kaggle loss.
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = epoch_loss/len(dataloader)
    print(f"Training loss: {train_loss:>7f}")
    return train_loss.item()


def validate(val_dataloader, model, loss_fn, device):
    num_batches = len(val_dataloader)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, Y in val_dataloader:
            X, Y = X.to(device), Y.to(device)
            Y = torch.flatten(Y, 1)
            Y_flat = torch.flatten(Y, 1)
            pred = model(X)
            pred_flat = torch.flatten(pred, 1)
            val_loss += loss_fn(pred_flat*255, Y_flat*255).item()
    val_loss /= num_batches
    print(f"Validation loss: {val_loss:>7f} ")
    print("-" * 50)
    return val_loss


if __name__=="__main__":
    main()