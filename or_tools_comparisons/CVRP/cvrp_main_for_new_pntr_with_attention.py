import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.DEPRECATED_CVRP_dataset import CapacitatedVehicleRoutingDataset, reward_fn

folder_name  = "DOT_VS_BAH"

from ploting.plot_utilities import plot_train_and_validation_loss, plot_train_and_validation_reward, plot_reward
from training_utilities.EarlyStopping import EarlyStopping

lr = 1e-2

def train_cvrp_model_pntr(actor, epochs, train_loader, validation_loader, experiment_details):
    optimizer = optim.Adam(actor.parameters(), lr=lr)


    # Metrics we want to show
    validation_loss_per_epoch = []
    train_loss_per_epoch = []
    validation_reward_per_epoch = []
    train_reward_per_epoch = []

    # Setting actor to training mode
    actor.train()

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



            # anti na pairnoume kateutheian to reward gia loss, pairnoume to reward- critic_estimate
            loss = torch.mean(reward.detach() * tour_logp.sum(dim=1))


            max_grad_norm =  2.# 1. #
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=max_grad_norm, norm_type=2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_batch_loss = loss.detach().data.item()
            current_batch_reward = torch.mean(reward.detach()).item()

            train_reward_at_epoch += current_batch_reward
            train_loss_at_epoch += current_batch_loss

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
        print(f'\nEpoch:{epoch}: Loss at epoch :{train_loss_at_epoch}, Reward at epoch:{train_reward_at_epoch}')

    plot_train_and_validation_loss(epoch, train_loss_per_epoch, validation_loss_per_epoch, experiment_details, folder_name)
    plot_train_and_validation_reward(epoch, train_reward_per_epoch, validation_reward_per_epoch, experiment_details, folder_name)

    return actor


if __name__ == '__main__':
    # testing purposes!!
    # epochs = 10
    # num_nodes = 4
    # train_size = 9#0
    # test_size = 10
    # batch_size = 3
    # STATIC_SIZE = 2
    # DYNAMIC_SIZE = 2
    # hidden_size = 128
    epochs = 10
    num_nodes = 4
    train_size = 250#0
    test_size = 100
    batch_size = 25#100
    STATIC_SIZE = 2
    DYNAMIC_SIZE = 2
    hidden_size = 256

    # TODO: add metric for time taken for training
    # TODO: add saving model
    experiment_details_for_dot = f'EXP1_WITHOUT_ACTOR_CRITIC_Dot_ATTENTION_ep={epochs}_nodes={num_nodes}_train_size={test_size}'
    experiment_details_for_bahdanau = f'EXP1_WITHOUT_ACTOR_CRITIC_Bahdanau_ATTENTION_ep={epochs}_nodes={num_nodes}_train_size={test_size}'

    from models.CVRP_SOLVER import CVRP_SOLVER_MODEL
    train_dataset = CapacitatedVehicleRoutingDataset(num_samples=train_size, input_size=num_nodes)
    test_dataset = CapacitatedVehicleRoutingDataset(num_samples=test_size, input_size=num_nodes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    actor = CVRP_SOLVER_MODEL(use_multihead_attention=False,
                              attention_type='Dot',
                              experiment_name = experiment_details_for_dot,
                              use_pointer_network=True,
                              use_pntr_with_attention_variations=True)

    train_cvrp_model_pntr(actor,
                          epochs,
                          train_loader,
                          validation_loader,
                          experiment_details_for_dot)

    # ### we can observe the results for 2 types of attention. We want to use the same datasets
    # # for our experiment to be meaningful
    # # BAHDANAY ATTENTION
    model_bahdanau_attention = CVRP_SOLVER_MODEL(use_multihead_attention=False,
                                            attention_type='Bahdanau',
                                            use_pointer_network=True,
                                            use_pntr_with_attention_variations=True)


    train_cvrp_model_pntr(model_bahdanau_attention,
                          epochs, train_loader,
                          validation_loader,
                          experiment_details_for_bahdanau)


