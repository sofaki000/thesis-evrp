import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import GVRPDataset, reward_func
from models.MHA_MODELS.MHA_model import MHA_EVRP_solver
from plot_utilities import plot_losses_and_rewards

epochs = 20
#model = EVRP_Solver()
model = MHA_EVRP_solver()

train_size = 100
num_nodes = 4
t_limit = 11
capacity = 60
num_afs = 3
max_grad = 2.
batch_size = 5
train_data = GVRPDataset(train_size, num_nodes, t_limit, capacity,num_afs)
train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
lr =0.001

optim = torch.optim.Adam(model.parameters(), lr=lr)
model.train()
losses_per_epochs = []
rewards_per_epochs = []

for epoch in range(epochs):
        reward_at_epoch = 0
        loss_at_epoch = 0
        for batch_id, (static, dynamic, distances) in enumerate(train_loader):
                probabilities, tours, tour_logp = model(static, dynamic,distances)
                rewards, _ = reward_func(tours, static, distances)
                loss = torch.mean(rewards.detach() * tour_logp.sum(dim=1))

                loss_at_epoch += loss.detach().item()
                reward_at_epoch += torch.mean(rewards.detach()).item()

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                optim.step()

        # epoch finished
        print(f'Epoch:{epoch}, loss:{loss_at_epoch}, reward:{reward_at_epoch}')
        losses_per_epochs.append(loss_at_epoch)
        rewards_per_epochs.append(reward_at_epoch)


# epochs finished
plot_losses_and_rewards(losses_per_epochs, rewards_per_epochs)