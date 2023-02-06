
from tqdm import tqdm
import torch
import torch.optim as optim
from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tsp_model import TSP_model
from dataset import  TSPDataset


def plot_losses(epoch, train_loss, val_loss):
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
    plt.savefig("train_and_val_loss.png")
    plt.clf()

def train_tsp_model(train_dataset, test_dataset):
    epochs = 20
    batch_size = 10
    num_nodes = 13


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    model = TSP_model(sequence_length=num_nodes)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    val_loss = []
    train_loss= []

    losses_per_epoch = []
    for epoch in range(epochs):
        loss_at_epoch = 0.0
        iterator = tqdm(train_loader, unit='Batch')

        for batch_id, sample_batch in enumerate(iterator):
            train_batch = Variable(sample_batch['Points'])
            target_batch = Variable(sample_batch['Solution'])

            loss,outputs = model(train_batch, target_batch)

            loss_at_epoch += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.data.item())

            if batch_id % 100 == 0:
                plot_losses(epoch, train_loss,val_loss)

            if batch_id % 100==0:
                model.eval()
                for val_batch in validation_loader:
                    train_batch = Variable(val_batch['Points'])
                    target_batch = Variable(val_batch['Solution'])

                    loss ,outputs = model(train_batch, target_batch)

                    val_loss.append(loss.data.item())

        # epoch finished
        print(f'Loss at epoch {epoch}: {loss_at_epoch}')
        losses_per_epoch.append(loss_at_epoch)


    # training finished
    plt.plot(losses_per_epoch)
    plt.savefig("losses.png")

    return model,outputs