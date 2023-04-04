import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.CVRP_dataset import reward_fn
from datasets.VRPTW_dataset import VRPTW_data, reward_fn_vrptw
from models.VRTW_SOLVER import VRPTW_SOLVER_MODEL

from plot_utilities import plot_train_and_validation_loss, plot_train_and_validation_reward


def train_vrptw_model(model, epochs, train_loader, validation_loader):
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

            static, dynamic, distance_matrix = sample_batch

            tour_indices, tour_logp, time_spent_at_each_route= model(static, dynamic, distance_matrix)

            reward = reward_fn_vrptw(static, tour_indices,time_spent_at_each_route)

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
                    static, dynamic, distance_matrix = sample_batch

                    validation_tour_indices, validation_tour_logp,time_spent_at_each_route = model(static, dynamic, distance_matrix)

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



def get_model_vrptw(use_pretrained_model,epochs, train_loader, validation_loader):
    model = VRPTW_SOLVER_MODEL()
    PATH = "trained_models/model_vrptw_conv1D_embeddings.pt"
    if use_pretrained_model:
        # load model from path
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(PATH))
    else:
        # train new model
        print("Training model...")
        model = train_vrptw_model(model, epochs, train_loader, validation_loader)
        torch.save(model.state_dict(), PATH)

    return model

if __name__ == '__main__':
    epochs = 40
    num_nodes = 5
    train_size = 1000
    test_size = 100
    batch_size = 25

    train_dataset = VRPTW_data(train_size, num_nodes)
    test_dataset = VRPTW_data(train_size, num_nodes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    model = VRPTW_SOLVER_MODEL( )
    model = train_vrptw_model(model, epochs, train_loader, validation_loader)