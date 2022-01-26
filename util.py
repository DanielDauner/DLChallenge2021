import torch
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np


def mse_loss(x, y):
    mse_loss = torch.nn.MSELoss(reduction="mean")
    pixel_wise_mse = mse_loss(x.reshape(-1),y.view(-1))
    return pixel_wise_mse


class VisualizeTraining():

    def __init__(self, out_dir, epochs):
        
        self.out_dir = out_dir
        self.epochs = epochs

        self.training_loss = []
        self.validation_loss = []
        self.start_time = 0.

    def update(self, train_loss, val_loss, current_epoch):
        
        self.training_loss.append(train_loss)
        self.validation_loss.append(val_loss)

        if current_epoch == 0:
            self.start_time = time.time()
        else:
            seconds_left = (time.time()-self.start_time)/(current_epoch+1) * (self.epochs-current_epoch)


        fig, ax = plt.subplots(1,1, figsize=(7.5,5))
        
        ax.plot(np.arange(current_epoch+1), self.training_loss, color="C3", label="Training Loss")
        ax.plot(np.arange(current_epoch+1), self.validation_loss, color="C2", label="Validation Loss")
        # You're a 
        ax.legend()
        
        if current_epoch != 0:
            ax.set_title(f"Time left: {str(datetime.timedelta(seconds=seconds_left))} Scores: {round(self.training_loss[-1],2)} / {round(self.validation_loss[-1],2)}")

        fig.savefig(self.out_dir)
        plt.close(fig)



        