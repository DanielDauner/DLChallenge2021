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
    model_g = models.BetaUNet2(beta=1).to(device)
    #model_g.load_state_dict(torch.load("./model/best.pth"))

    model_d = models.Discriminator().to(device)



    optimizer_g = torch.optim.Adam(model_g.parameters(), learning_rate)
    optimizer_d = torch.optim.Adam(model_d.parameters(), learning_rate)


    #optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9)
    lr_scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, [200, 350], gamma=0.1)
    lr_scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, [200, 350], gamma=0.1)

    Visualize = VisualizeTraining(f"training_{args.adversarial_weight}.png", epochs)
    bce_loss = torch.nn.BCELoss(reduction='sum')
   

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        print("-" * 50)
        start = time.time()
        train_loss = train_one_epoch(train_dataloader, model_g, model_d, mse_loss, bce_loss, optimizer_g, optimizer_d, device, args.adversarial_weight)
        
        print("Time per Epoch: ", time.time()-start)

        val_loss = validate(val_dataloader, model_g, mse_loss, device)
        Visualize.update(train_loss, val_loss, epoch)

        lr_scheduler_d.step()
        lr_scheduler_g.step()

        sys.stdout.flush()
        sys.stderr.flush()

    if args.model_path:
        torch.save(model_g.state_dict(), f"./model/model_{args.adversarial_weight}.pth")
        print(f"Sucesfully saved model to {args.model_path}")    

    print("Done!")


def train_one_epoch(dataloader, model_g, model_d, mse_loss, bce_loss, optimizer_g, optimizer_d, device, adversarial_weight):
    model_g.train()
    model_d.train()
    epoch_loss_d = 0.0
    epoch_loss_g = 0.0
    epoch_loss_gr = 0.0
    epoch_loss_ga = 0.0

    epoch_loss_bce = 0.0



    for batch_number, (X, Y) in enumerate(dataloader):

        X, Y = X.to(device), Y.to(device)


        valid = torch.ones((X.shape[0],1), dtype=torch.float32, requires_grad=False, device=device)
        fake = torch.zeros((X.shape[0],1), dtype=torch.float32, requires_grad=False, device=device)

        # (3) Train model_g on all-fake data
        model_g.train()
        model_d.eval()

        optimizer_g.zero_grad()

        fake_Y, mu, var = model_g(X)
        with torch.no_grad():
            out_d_Y, feature_1_Y, feature_2_Y = model_d(Y, return_features=True)
            out_d_fake_Y, feature_1_fake_Y, feature_2_fake_Y = model_d(fake_Y, return_features=True)

        epoch_loss_bce += bce_loss(out_d_fake_Y, valid).item()

        # adversarial_loss_g = bce_loss(output_d, valid)
        adversarial_loss_g = (torch.nn.functional.mse_loss(feature_1_Y, feature_1_fake_Y, reduction="sum")+torch.nn.functional.mse_loss(feature_2_Y, feature_2_fake_Y, reduction="sum"))/(2*X.shape[0])
        epoch_loss_ga += adversarial_loss_g.item()

        reconstruction_loss,_,__ = model_g.loss(fake_Y, Y, mu, var)
        epoch_loss_gr += reconstruction_loss.item()
        
        err_g = reconstruction_loss+adversarial_loss_g
        epoch_loss_g += err_g.item()
        
        err_g.backward()
        optimizer_g.step()

        


        # (1) Train model_d on all-real data
        model_g.eval()
        model_d.train()

        optimizer_d.zero_grad()

        output_d_real = model_d(Y)
        err_d_real = bce_loss(output_d_real, valid)
        

        # (2) Train model_d on all-fake data
        with torch.no_grad():
            fake_Y,_,__ = model_g(X)
        
        output_d_fake = model_d(fake_Y)
        err_d_fake = bce_loss(output_d_fake, fake)
        err_d = (err_d_fake+err_d_real)/2.

        err_d.backward()
        epoch_loss_d += err_d.item()

        optimizer_d.step()

    train_loss_d = epoch_loss_d/len(dataloader)
    train_loss_g = epoch_loss_g/len(dataloader)
    train_loss_ga = epoch_loss_ga/len(dataloader)
    train_loss_gr = epoch_loss_gr/len(dataloader)

    epoch_loss_bce = epoch_loss_bce/len(dataloader)


    print(f"Training loss D: {train_loss_d:>7f}, G: {train_loss_g:>7f} (R: {train_loss_gr:>7f}, A:{train_loss_ga:>7f})")
    print(f"BCE Loss: {epoch_loss_bce:>7f}")
    return train_loss_g




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
    print(f"Validation loss: {val_loss:>7f} ")
    print("-" * 50)
    return val_loss


if __name__=="__main__":
    main()