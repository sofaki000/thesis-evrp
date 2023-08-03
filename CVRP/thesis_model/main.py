import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from CVRP.thesis_model.model import PointerNet
from datasets.CVRP_dataset import CapacitatedVehicleRoutingDataset, reward_fn
from models.CVRP_Critic import StateCritic

from ploting.plot_utilities import plot_train_and_validation_loss, plot_train_and_validation_reward, plot_reward, \
    plot_train_and_validation_distance
from training_utilities.EarlyStopping import EarlyStopping
from training_utilities.SavingAndLoadingUtilities import saveModelToPath

STATIC_SIZE = 2
DYNAMIC_SIZE = 2
f = open("training_metrics.txt", "a")

def train_cvrp_model_pntr(actor,
                          critic,
                          epochs,
                          train_loader,
                          validation_loader,
                          experiment_details,
                          lr,
                          folder_name):
    optimizer = optim.Adam(actor.parameters(), lr=lr)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr)


    # Metrics we want to show
    validation_loss_per_epoch = []
    train_loss_per_epoch = []
    validation_reward_per_epoch = []
    critic_rewards_per_epoch = []
    critic_losses_per_epoch = []
    train_reward_per_epoch = []

    # Setting actor and critic to training mode
    actor.train()
    critic.train()

    # Training with early stopping
    early_stopping = EarlyStopping()

    num_train_batches = len(train_loader)
    num_val_batches = len(validation_loader)

    for epoch in range(epochs):


        iterator = tqdm(train_loader, unit='Batch')
        train_loss_at_epoch = 0.0
        train_reward_at_epoch = 0.0

        validation_loss_at_epoch = 0.0
        validation_reward_at_epoch = 0.0

        for batch_id, sample_batch in enumerate(iterator):

            static, dynamic, x0 = sample_batch

            tour_indices, tour_logp = actor(static, dynamic, epoch)

            reward = reward_fn(static, tour_indices)

            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic).view(-1)

            # anti na pairnoume kateutheian to reward gia loss, pairnoume to reward- critic_estimate
            # loss = torch.mean(reward.detach() * tour_logp.sum(dim=1))
            advantage = (reward - critic_est)
            loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))

            max_grad_norm = 1. #2.
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=max_grad_norm, norm_type=2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_batch_loss = loss.detach().data.item()
            current_batch_reward = torch.mean(reward.detach()).item()

            train_reward_at_epoch += current_batch_reward
            train_loss_at_epoch += current_batch_loss


            ### for the critic
            critic_loss = torch.mean(advantage ** 2)

            optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            optimizer_critic.step()

            critic_rewards_per_epoch.append(torch.mean(critic_est.detach()).item())
            critic_losses_per_epoch.append(torch.mean(critic_loss.detach().item()))
            actor.eval()

            for sample_batch  in validation_loader:
                static, dynamic, x0 = sample_batch
                validation_tour_indices, validation_tour_logp = actor(static, dynamic)
                validation_reward = reward_fn(static, validation_tour_indices)
                validation_loss = torch.mean(validation_reward.detach() * validation_tour_logp.sum(dim=1))

                validation_loss_at_epoch+= validation_loss.data.item()
                validation_reward_at_epoch += torch.mean(validation_reward.detach()).item()


            actor.train()


        # END OF EPOCH
        train_loss_per_epoch.append(train_loss_at_epoch//num_train_batches)
        train_reward_per_epoch.append(train_reward_at_epoch//num_train_batches)

        validation_loss_per_epoch.append(validation_loss_at_epoch// num_val_batches)
        validation_reward_per_epoch.append(validation_reward_at_epoch // num_val_batches)

        early_stopping(validation_loss, actor, path="checkpoints\\model")
        if early_stopping.early_stop:
            print(f"-------------------Early stopping at epoch {epoch}-------------------")
            break

        # epoch finished
        print(f'\nEpoch:{epoch}: Loss :{train_loss_at_epoch}, Reward :{train_reward_at_epoch}')


    plot_train_and_validation_loss(epoch, train_loss_per_epoch,
                                   validation_loss_per_epoch,
                                   experiment_details,
                                   folder_name)

    plot_train_and_validation_distance(epoch,
                                     train_reward_per_epoch,
                                     validation_reward_per_epoch,
                                     experiment_details,
                                     folder_name)

    plot_reward(critic_rewards_per_epoch,
                "Critic rewards",
                f'critic_rewards_{experiment_details}',folder_name)

    plot_reward(critic_losses_per_epoch,
                "Critic losses",
                f'critic_losses_{experiment_details}', folder_name)

    saveModelToPath(actor, f"models\\{experiment_details}")

    return actor




def run_actor_critic_dot_vs_additive(epochs,num_nodes ,
                                     train_size ,
                                     test_size ,
                                     batch_size ,
                                     hidden_size,
                                     lr,
                                     folder_name):

    experiment_name = f'BIGGER_SIZE_Dcd_inp_ep={epochs}_hz={hidden_size}_nodes={num_nodes}_train_size={test_size}_lr={lr}'

    f.write(f"Experiment name: {experiment_name}\n")

    train_dataset = CapacitatedVehicleRoutingDataset(num_samples=train_size, input_size=num_nodes)
    test_dataset = CapacitatedVehicleRoutingDataset(num_samples=test_size, input_size=num_nodes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)


    model_bahdanau_attention = PointerNet(embedding_size=hidden_size,
                                    hidden_size=hidden_size,
                                    experiment_name=experiment_name,
                                    experiment_folder_name="metrics")

    critic_bahdanau_attention = StateCritic(STATIC_SIZE, DYNAMIC_SIZE,  hidden_size)

    import time
    training_start_time = time.time()
    train_cvrp_model_pntr(model_bahdanau_attention,
                          critic_bahdanau_attention,
                          epochs, train_loader,
                          validation_loader,
                          experiment_name,
                          lr,
                          folder_name)

    content = '\n\nTraining finished, took {:.3f}s'.format(time.time() - training_start_time)
    print(content )
    f.write(content)
    f.write(f'\nExperiment name: {experiment_name}\n')

#TODO: add qury_dynamic_embedding_step to overfleaf + add the last experiment u will have the results for 200 nodes
if __name__ == '__main__':
    ## TEST RUN
    # epochs = 2   # 5
    # num_nodes = 4  # 0
    # train_size = 100  # 0
    # test_size = 100  # 0
    # batch_size = 25  # 0
    # hidden_size = 256
    # lr = 1e-4

    epochs = 20#5
    num_nodes = 4
    train_size = 1000#0
    test_size = 300#0
    batch_size = 25#0
    hidden_size = 256
    lr = 1e-4

    exp_run_folder="DE_AT_DEC_IN_BIGGER_TRAIN_SIZE"
    folder_name = f"{exp_run_folder}//4cust_dynamic_embedding_at_dc_in"
    f.write(f"Apotelesmata sto folder: {folder_name}\n")

    run_actor_critic_dot_vs_additive(epochs,
                                     num_nodes,
                                     train_size,
                                     test_size,
                                     batch_size,
                                     hidden_size,
                                     lr,
                                     folder_name)
    ##################
    epochs = 20#5
    num_nodes = 200#0
    train_size = 1000#0
    test_size = 300#0
    batch_size = 25#0
    hidden_size = 256
    lr = 1e-4

    exp_run_folder="DE_AT_DEC_IN_BIGGER_TRAIN_SIZE"
    folder_name = f"{exp_run_folder}//200cust_dynamic_embedding_at_dc_in"
    f.write(f"Apotelesmata sto folder: {folder_name}\n")

    run_actor_critic_dot_vs_additive(epochs,
                                     num_nodes,
                                     train_size,
                                     test_size,
                                     batch_size,
                                     hidden_size,
                                     lr,
                                     folder_name)

    epochs = 20  # 5
    num_nodes = 40  # 0
    train_size = 1000  # 0
    test_size = 300  # 0
    batch_size = 25  # 0
    hidden_size = 256
    lr = 1e-4

    folder_name = f"{exp_run_folder}//40cust_dynamic_embedding_at_dc_in"
    f.write(f"Apotelesmata sto folder: {folder_name}\n")

    run_actor_critic_dot_vs_additive(epochs,
                                     num_nodes,
                                     train_size,
                                     test_size,
                                     batch_size,
                                     hidden_size,
                                     lr,
                                     folder_name)

    epochs = 20  # 5
    num_nodes = 100  # 0
    train_size = 1000  # 0
    test_size = 300  # 0
    batch_size = 25  # 0
    hidden_size = 256
    lr = 1e-4

    folder_name = f"{exp_run_folder}//100cust_dynamic_embedding_at_dc_in"
    f.write(f"Apotelesmata sto folder: {folder_name}\n")

    run_actor_critic_dot_vs_additive(epochs,
                                     num_nodes,
                                     train_size,
                                     test_size,
                                     batch_size,
                                     hidden_size,
                                     lr,
                                     folder_name)
    f.close()
    # epochs = 25
    # num_nodes = 100
    # train_size = 1000
    # test_size = 400
    # batch_size = 100
    # hidden_size = 256
    #
    # run_actor_critic_dot_vs_additive(epochs, num_nodes, train_size, test_size, batch_size, hidden_size)
    #
    # epochs = 25
    # num_nodes = 200
    # train_size = 1000
    # test_size = 400
    # batch_size = 100
    # hidden_size = 256
    #
    # run_actor_critic_dot_vs_additive(epochs, num_nodes, train_size, test_size, batch_size, hidden_size)
