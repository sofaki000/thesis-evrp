import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from EVRP.evrp_config import evrp_losses_dir, evrp_rewards_dir
from datasets.EVRP_dataset import GVRPDataset, reward_func
from models.EVRP_SOLVER import EVRP_SOLVER
from ploting.plot_utilities import get_filename_time, plot_train_and_validation_loss, plot_train_and_validation_reward
from training_utilities.EarlyStopping import EarlyStopping



def run_experiment_with_config(epochs, train_size, num_nodes,batch_size):
    #epochs =  20
    #model = EVRP_Solver()
    model = EVRP_SOLVER() #MHA_EVRP_solver()

    #train_size = 300
    # num_nodes = 4
    t_limit = 11
    capacity = 60
    num_afs = 3
    max_grad = 2.
    train_data = GVRPDataset(train_size, num_nodes, t_limit, capacity,num_afs)
    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)

    validation_data = GVRPDataset(train_size, num_nodes, t_limit, capacity,num_afs)
    validation_loader= DataLoader(validation_data, batch_size, True, num_workers=0)
    lr =0.001

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses_per_epochs = []
    rewards_per_epochs = []
    validation_losses_per_epochs = []
    validation_rewards_per_epochs = []
    early_stopping = EarlyStopping()
    num_train_batches = len(train_loader)
    num_val_batches = len(validation_loader)
    experiment_details = f'more_nodes_ep={epochs}_nodes={num_nodes}_train_size={train_size}_bs={batch_size}'

    for epoch in range(epochs):
            reward_at_epoch = 0
            loss_at_epoch = 0
            validation_loss_at_epoch = 0
            validation_reward_at_epoch = 0

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

                    model.eval()

                    for batch_id, (static, dynamic, distances) in enumerate(validation_loader):
                        probabilities, tours, tour_logp = model(static, dynamic,distances)
                        validation_reward, _ = reward_func(tours, static, distances)
                        validation_loss = torch.mean(validation_reward.detach() * tour_logp.sum(dim=1))

                        validation_loss_at_epoch += validation_loss.data.item()
                        validation_reward_at_epoch += torch.mean(validation_reward.detach()).item()


                    model.train()

            early_stopping(validation_loss, model, path=f"checkpoints\\{experiment_details}")
            if early_stopping.early_stop:
                print(f"-------------------Early stopping at epoch {epoch}-------------------")
                break

            # epoch finished
            print(f'Epoch:{epoch}, loss:{loss_at_epoch}, reward:{reward_at_epoch}')
            losses_per_epochs.append(loss_at_epoch)
            rewards_per_epochs.append(reward_at_epoch)

            validation_losses_per_epochs.append(validation_loss_at_epoch)
            validation_rewards_per_epochs.append(validation_reward_at_epoch)


    def plot_losses_and_rewards(losses_per_epochs, rewards_per_epochs):
        time = get_filename_time()
        # epochs finished
        plt.plot(losses_per_epochs)
        plt.savefig(f"{evrp_losses_dir}losses{time}.png")
        plt.clf()
        plt.plot(rewards_per_epochs)
        plt.savefig(f"{evrp_rewards_dir}rewards{time}.png")

    # epochs finished
    plot_losses_and_rewards(losses_per_epochs, rewards_per_epochs)


    folder_name = "metrics"
    plot_train_and_validation_loss(epoch, losses_per_epochs, validation_losses_per_epochs, experiment_details, folder_name)
    plot_train_and_validation_reward(epoch, rewards_per_epochs, validation_rewards_per_epochs, experiment_details,
                                     folder_name)



## TEST:
# try:
#     epochs = 3
#     train_size =20
#     num_nodes = 4
#     batch_size = 10
#     run_experiment_with_config(epochs, train_size, num_nodes,batch_size)
# except:
#     print("an error occured")
#
try:
    epochs = 20
    train_size =200
    num_nodes = 100
    batch_size = 10
    run_experiment_with_config(epochs, train_size, num_nodes,batch_size)
except:
    print("an error occured")


try:
    epochs = 20
    train_size = 400
    num_nodes = 100
    batch_size = 25
    run_experiment_with_config(epochs, train_size, num_nodes,batch_size)
except:
    print("an error occured")

try:
    epochs = 20
    train_size = 200
    num_nodes = 400
    batch_size = 25
    run_experiment_with_config(epochs, train_size, num_nodes,batch_size)
except:
    print("an error occured")

try:
    epochs = 20
    train_size = 400
    num_nodes = 400
    batch_size = 25
    run_experiment_with_config(epochs, train_size, num_nodes,batch_size)
except:
    print("an error occured")



#TODO: tsekare results apo actor critic main + se auto to main ta 4 experiments ti eginan