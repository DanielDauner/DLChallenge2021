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



# torch.backends.cudnn.deterministic = True

def main():

    args = argument_handling.parse_args()
    train_dataloader, val_dataloader, test_dataloader = \
        data_handling.get_dataloaders(args.batch_size,args.validation_fraction,
                                      args.train_data_path,args.train_labels_path,args.test_data_path)

    
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    learning_rate = args.learning_rate
    epochs = args.epochs
    bootstrap = args.bootstrap
    save_model_epoch = 1
    save_model_path = f"./model/epoch/"
 
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")
    
    # model = models.BetaUNet2(beta=args.beta).to(device)
    model = models.BetaUNet2Large(beta=args.beta).to(device)
    model.load_state_dict(torch.load("./model/model_pretrain.pth"))
    
    # TODO: enable pretraining
    #if args.pretrain_path:

    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40], gamma=0.1)

    Visualize = VisualizeTraining(f"/home/daniel/Dropbox/DeepLearningResults/training.png", epochs)
   

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        print("-" * 50)
        start = time.time()
        train_loss = train_one_epoch(train_dataloader, model, mse_loss, optimizer, device)
        
        print("Time per Epoch: ", time.time()-start)

        print(f"Training mse:\t {train_loss:>7f}")
        
        val_loss = validate(val_dataloader, model, mse_loss, device)
        print(f"Validation mse:\t {val_loss:>7f}")
        print("-" * 50)

        Visualize.update(train_loss, val_loss, epoch)

        sys.stdout.flush()
        sys.stderr.flush()
        lr_scheduler.step()

        if epoch % save_model_epoch == 0:
            torch.save(model.state_dict(), os.path.join(save_model_path, f"model_{epoch + 1}_{int(val_loss)}.pth"))
            print(f"Sucesfully saved model at epoch {epoch + 1}")    

    
    if args.model_path:
        torch.save(model.state_dict(), args.model_path)
        print(f"Sucesfully saved model to {args.model_path}")    

    print("Done!")



def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    epoch_mse_loss = 0.0
    for batch_number, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)

        pred, mu, logvar = model(X)
        loss, rec, kl = model.loss(pred, Y, mu, logvar)

        epoch_mse_loss += loss_fn(torch.flatten(pred, 1)*255, torch.flatten(Y, 1)*255).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return epoch_mse_loss/len(dataloader)



def validate(val_dataloader, model, loss_fn, device):
    num_batches = len(val_dataloader)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, Y in val_dataloader:
            X, Y = X.to(device), Y.to(device)
            Y = torch.flatten(Y, 1)
            Y_flat = torch.flatten(Y, 1)
            pred,_,__ = model(X)
            pred_flat = torch.flatten(pred, 1)

            val_loss += loss_fn(pred_flat*255, Y_flat*255).item()
    val_loss /= num_batches
    
    return val_loss


if __name__=="__main__":
    main()