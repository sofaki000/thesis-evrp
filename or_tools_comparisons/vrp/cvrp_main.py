import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.CVRP_dataset import CapacitatedVehicleRoutingDataset, reward_fn

from plot_utilities import plot_train_and_validation_loss, plot_train_and_validation_reward




def train_cvrp_model_pntr(model, epochs, train_loader, validation_loader):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    val_loss = []
    train_loss = []
    val_reward = []
    train_reward = []

    for epoch in range(epochs):
        iterator = tqdm(train_loader, unit='Batch')
        loss_at_epoch = 0.0
        reward_at_epoch = 0.0

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
            current_reward = torch.mean(reward.detach()).item()
            train_reward.append(current_reward)

            train_loss.append(current_batch_loss)
            reward_at_epoch += current_reward
            loss_at_epoch += current_batch_loss

            if batch_id % 100 == 0:
                model.eval()
                for sample_batch  in validation_loader:
                    static, dynamic, x0 = sample_batch
                    validation_tour_indices, validation_tour_logp = model(static, dynamic)
                    validation_reward = reward_fn(static, validation_tour_indices)
                    validation_loss = torch.mean(validation_reward.detach() * validation_tour_logp.sum(dim=1))
                    val_loss.append(validation_loss.data.item())
                    val_reward.append(torch.mean(validation_reward.detach()).item())

            if batch_id % 100 == 0  and epoch== epochs-1:
                test_size = len(train_loader.dataset)
                num_nodes = static.size(2)
                experiment_details = f'ep={epochs}_nodes={num_nodes}_train_size={test_size}'
                plot_train_and_validation_loss(epoch, train_loss, val_loss,experiment_details)
                plot_train_and_validation_reward(epoch, train_reward, val_reward,experiment_details)

        # epoch finished
        print(f'Epoch:{epoch}: Loss:{loss_at_epoch}, Reward:{reward_at_epoch}')


    return model


if __name__ == '__main__':
    epochs = 10
    num_nodes = 4
    train_size = 100
    test_size = 10
    batch_size = 25
    from models.CVRP_SOLVER import CVRP_SOLVER_MODEL
    train_dataset = CapacitatedVehicleRoutingDataset(num_samples=train_size, input_size=num_nodes)
    test_dataset = CapacitatedVehicleRoutingDataset(num_samples=test_size, input_size=num_nodes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    model = CVRP_SOLVER_MODEL(use_multihead_attention=False, use_pointer_network=True)
    train_cvrp_model_pntr(model, epochs, train_loader, validation_loader)