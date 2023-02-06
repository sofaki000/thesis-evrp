import math
from model import PointerNet
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from IPython.display import clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt

class SortDataset(Dataset):

    def __init__(self, data_len, num_samples, random_seed=111):
        super(SortDataset, self).__init__()
        torch.manual_seed(random_seed)

        self.data_set = []
        for _ in tqdm(range(num_samples)):
            x = x = torch.randperm(data_len)
            self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]

def plot_losses(epoch, train_loss,val_loss):
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('train epoch %s loss %s' % (epoch, train_loss[-1] if len(train_loss) else 'collecting'))
        plt.plot(train_loss)
        plt.grid()
        plt.subplot(132)
        plt.title('val epoch %s loss %s' % (epoch, val_loss[-1] if len(val_loss) else 'collecting'))
        plt.plot(val_loss)
        plt.grid()
        plt.savefig("result.png")

if __name__ == '__main__':
    train_size = 10000
    val_size = 100
    train_dataset = SortDataset(10, train_size)
    val_dataset   = SortDataset(10, val_size)

    pointer = PointerNet(embedding_size=32, hidden_size=32, seq_len=10, n_glimpses=1, tanh_exploration=10, use_tanh=True)
    adam = optim.Adam(pointer.parameters(), lr=1e-4)

    n_epochs = 1
    train_loss = []
    val_loss = []
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)

    for epoch in range(n_epochs):
        for batch_id, sample_batch in enumerate(train_loader):

            inputs = Variable(sample_batch)
            target = Variable(torch.sort(sample_batch)[0])

            loss = pointer(inputs, target)

            adam.zero_grad()
            loss.backward()
            adam.step()

            train_loss.append(loss.data.item()) #loss.data[0])

            if batch_id % 10 == 0:
                plot_losses(epoch, train_loss,val_loss)

            if batch_id % 100 == 0:
                pointer.eval()
                for val_batch in val_loader:
                    inputs = Variable(val_batch)
                    target = Variable(torch.sort(val_batch)[0])

                    loss = pointer(inputs, target)
                    val_loss.append(loss.data.item())#loss.data[0])