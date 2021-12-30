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
        data_handling.get_dataloaders(args.batch_size,args.validation_fraction,
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
    model_g = models.UNet().to(device)
    # model_g.load_state_dict(torch.load("./model/unet_train.pth"))

    model_d = models.Discriminator().to(device)



    optimizer_g = torch.optim.Adam(model_g.parameters(), learning_rate)
    optimizer_d = torch.optim.Adam(model_d.parameters(), learning_rate)


    #optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [], gamma=0.1)

    Visualize = VisualizeTraining("training.png", epochs)
    bce_loss = torch.nn.BCELoss()
   

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        print("-" * 50)
        start = time.time()
        train_loss = train_one_epoch(train_dataloader, model_g, model_d, mse_loss, bce_loss, optimizer_g, optimizer_d, device)
        
        print("Time per Epoch: ", time.time()-start)

        val_loss = validate(val_dataloader, model_g, mse_loss, device)
        Visualize.update(train_loss, val_loss, epoch)

        sys.stdout.flush()
        sys.stderr.flush()
    
    if args.model_path:
        torch.save(model_g.state_dict(), args.model_path)
        print(f"Sucesfully saved model to {args.model_path}")    

    print("Done!")


def train_one_epoch(dataloader, model_g, model_d, mse_loss, bce_loss, optimizer_g, optimizer_d, device):
    model_g.train()
    model_d.train()
    epoch_loss_d = 0.0
    epoch_loss_g = 0.0
    

    for batch_number, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        optimizer_d.zero_grad()
        
        # (1) Train model_d on all-real data
        model_g.eval()
        model_d.train()

        model_d.zero_grad()
        output_d_real = model_d(Y)
        
        real_label = torch.ones(output_d_real.shape, device=device)
        fake_label = torch.zeros(output_d_real.shape, device=device)
        
        err_d_real = bce_loss(output_d_real, real_label)
        err_d_real.backward()

        # (2) Train model_d on all-fake data
        
        fake_Y = model_g(X)
        output_d_fake = model_d(fake_Y)
        err_d_fake = bce_loss(output_d_fake, fake_label)
        err_d = (err_d_fake+err_d_real)/2.

        err_d_fake.backward()
        epoch_loss_d += err_d

        optimizer_d.step()

        # (3) Train model_g on all-fake data
        _, real_features = model_g(Y,return_feature=True)
        model_g.train()
        model_d.eval()

        optimizer_g.zero_grad()

        model_g.zero_grad()
        fake_Y, fake_features = model_g(X,return_feature=True)
        output_d = model_d(fake_Y)
        err_g = bce_loss(output_d, real_label)\
            +1000*mse_loss(torch.flatten(fake_Y, 1), torch.flatten(Y, 1))\
                +mse_loss(torch.flatten(fake_features, 1), torch.flatten(real_features, 1))
        epoch_loss_g += err_g
        
        err_g.backward()
        optimizer_g.step()

    train_loss_d = epoch_loss_d/len(dataloader)
    train_loss_g = epoch_loss_g/len(dataloader)
    print(f"Training loss D:{train_loss_d:>7f}, G:{train_loss_g:>7f}")
    return train_loss_g.item()

# def train_d(optimizer_d, model_d, Y, Y_fake):
#     # (1) Train model_d on all-real data
#     #model_d.zero_grad()
#     output_d_real = model_d(Y)
    
#     real_label = torch.zeros(output_d_real.shape, device=device)
#     fake_label = torch.ones(output_d_real.shape, device=device)
    
#     err_d_real = bce_loss(output_d_real, real_label)
#     #err_d_real.backward(retain_graph=True)
#     err_d.backward()
#     fake_Y = model_g(X)
#     output_d_fake = model_d(fake_Y)
#     err_d_fake = bce_loss(output_d_fake, fake_label)
#     err_d = err_d_fake+err_d_real

#     err_d.backward()
#     epoch_loss_d += err_d

#     optimizer_d.step()

#     # (2) Train model_g on all-fake data
#     #model_g.zero_grad()
#     output_d = model_d(fake_Y)
#     err_g = bce_loss(output_d, real_label)\
#                 + mse_loss(torch.flatten(fake_Y, 1), torch.flatten(Y, 1))
#     epoch_loss_g += err_g




# def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
#     model.train()
#     epoch_loss= 0.0
#     for batch_number, (X, Y) in enumerate(dataloader):
#         X, Y = X.to(device), Y.to(device)

#         Y_flat = torch.flatten(Y,1)
#         pred = model(X)
#         pred_flat = torch.flatten(pred, 1)
        
#         loss = loss_fn(pred_flat*255, Y_flat*255) # to be consistent with the kaggle loss.
#         epoch_loss += loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     train_loss = epoch_loss/len(dataloader)
#     print(f"Training loss: {train_loss:>7f}")
#     return train_loss.item()


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