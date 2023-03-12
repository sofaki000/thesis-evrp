import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.capacitated_vrp_dataset import CapacitatedVehicleRoutingDataset, reward_fn

from plot_utilities import plot_train_and_validation_loss, plot_train_and_validation_reward, create_distance_matrix, \
    create_distance_matrix_for_batch_elements


def train_model_with_multihead_attention(model, epochs, train_loader, validation_loader):
    max_grad = 2.

    lr = 0.00000001
    optimizer = optim.Adam(model.parameters(), lr=lr)
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

            distance_matrix = create_distance_matrix_for_batch_elements(static)

            outputs, tours, tour_logp  = model(static, dynamic,distance_matrix)

            reward = reward_fn(static, tours)
            loss = torch.mean(reward.detach() * tour_logp.sum(dim=1))

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_grad)
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
                    distance_matrix = create_distance_matrix_for_batch_elements(static)
                    outputs, validation_tour_indices, validation_tour_logp  = model(static, dynamic,distance_matrix)
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
    from models.CVRP_SOLVER import CVRP_SOLVER_MODEL
    epochs = 10
    num_nodes = 13 # THELEI POLLA NODES ALLIWS LEADS TO NAN!!!
    train_size = 100
    test_size = 100
    batch_size = 15
    torch.autograd.set_detect_anomaly(True)
    train_dataset = CapacitatedVehicleRoutingDataset(num_samples=train_size, input_size=num_nodes)
    test_dataset = CapacitatedVehicleRoutingDataset(num_samples=test_size, input_size=num_nodes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    model = CVRP_SOLVER_MODEL(use_multihead_attention=True, use_pointer_network=False)

    trained_model = train_model_with_multihead_attention(model, epochs, train_loader, validation_loader)