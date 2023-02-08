import torch
import torch.nn as nn
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets.capacitated_vrp_dataset import CapacitatedVehicleRoutingDataset, reward_fn
from plot_utilities import get_filename_time
from cvrp_model import CVRPSolver


model = CVRPSolver()

def plot_train_and_validation_loss(epoch, train_loss, val_loss):
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
    plt.savefig(f"train_and_val_loss{get_filename_time()}.png")
    plt.clf()

def train_cvrp_model(epochs,train_loader ,validation_loader):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    val_loss = []
    train_loss = []

    for epoch in range(epochs):
        iterator = tqdm(train_loader, unit='Batch')
        loss_at_epoch = 0.0

        for batch_id, sample_batch in enumerate(iterator):

            static, dynamic, x0 = sample_batch

            tour_indices, tour_logp = model(static, dynamic)

            reward = reward_fn(static, tour_indices)
            loss = torch.mean(reward.detach() * tour_logp.sum(dim=1))

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1., norm_type=2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_batch_loss = loss.detach().data.item()
            train_loss.append(current_batch_loss)
            loss_at_epoch += current_batch_loss

            if batch_id % 100 == 0:
                model.eval()
                for sample_batch  in validation_loader:
                    static, dynamic, x0 = sample_batch
                    validation_tour_indices, validation_tour_logp = model(static, dynamic)
                    validation_reward = reward_fn(static, validation_tour_indices)
                    validation_loss = torch.mean(validation_reward.detach() * validation_tour_logp.sum(dim=1))
                    val_loss.append(validation_loss.data.item())

            if batch_id % 100 == 0  and epoch== epochs-1:
                plot_train_and_validation_loss(epoch, train_loss, val_loss)

        # epoch finished
        print(f'Epoch:{epoch}: Loss:{loss_at_epoch}')

    return model


if __name__ == '__main__':
    epochs = 10
    num_nodes = 4
    train_size = 100
    test_size = 10
    batch_size = 25

    train_dataset = CapacitatedVehicleRoutingDataset(num_samples=train_size, input_size=num_nodes)
    test_dataset = CapacitatedVehicleRoutingDataset(num_samples=test_size, input_size=num_nodes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    train_cvrp_model(epochs, train_loader ,validation_loader)