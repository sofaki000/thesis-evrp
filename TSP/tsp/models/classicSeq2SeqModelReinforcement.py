import numpy as np
from scipy.spatial import distance_matrix
import torch.nn.functional as F
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets.TSP_dataset import TSPDataset, TSPDatasetWithoutSolutions
from ploting.metrics.plot_average_tour_length import plot_average_tour_length
from ploting.metrics.plot_tour import format_tours, print_tensor, format_tour
from ploting.plot_utilities import plot_train_and_validation_loss

static_features = 2
hidden_size = 128

# experiments: arxika stohastic kai deterministic me argmax thing
# twra tha dokimasw na mhn dinw ta targets ston decoder ws input, alla to prohgoumeno input
# pou tha dinetai mesw enos embedding ofc

## TODO: fix bug that outputs the same route all the time
# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.RNN(input_size=static_features,
#                               hidden_size=hidden_size,
#                               batch_first=True)
#
#     def forward(self, input):
#         '''
#         input: [batch_size, num_nodes (sequence_length), static_features]
#         '''
#         return self.encoder(input)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        dropout_p = 0.1
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(static_features, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, sequence_length):
        super(Decoder, self).__init__()

        self.embedding = nn.Linear(1, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.out = nn.Linear(hidden_size, sequence_length)

    def apply_mask_to_logits(self, logits, mask, indexes):
            batch_size = logits.size(0)
            clone_mask = mask.clone()

            if indexes is not None:
                clone_mask[[i for i in range(batch_size)], indexes.data.squeeze(1).long()] = 1
                logits[clone_mask.unsqueeze(1)] = -np.inf
            else:
                logits[:, :] = -np.inf
                # we want to start from depot, ie the first node
                logits[:,:,0] = 1
            return logits, clone_mask

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        seq_len =  encoder_outputs.size(1)

        decoder_input = torch.ones(batch_size, 1)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        tours = []
        tour_logp = []
        mask = torch.zeros(batch_size, seq_len).byte()

        chosen_indexes = None
        for i in range(seq_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach().float()  # detach from history as input


            masked_logits, mask = self.apply_mask_to_logits(decoder_output, mask, chosen_indexes)

            # We transform decoder output to the actual result
            chosen_indexes = torch.argmax(masked_logits, dim=2).float()  # [batch_size, 1]
            log_probs = F.log_softmax(decoder_output, dim=2)
            logp = torch.gather(log_probs, 2, chosen_indexes.unsqueeze(2).long()).squeeze(2) #batch_size, 1

            tour_logp.append(logp.unsqueeze(1))
            tours.append(chosen_indexes.unsqueeze(1))

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        tours = torch.cat(tours, 2)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tours, tour_logp

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output.unsqueeze(1), hidden)
        output = self.out(output)
        return output, hidden

def reward_fn(static, tour_indices):
    """
    static: [batch_size, 2, sequence_length]
    tour_indices: [batch_size, tour_length]
    Euclidean distance between all cities / nodes given by tour_indices
    """
    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Ensure we're always returning to the depot - note the extra concat
    # won't add any extra loss, as the euclidean distance between consecutive
    # points is 0 # i was ensuring this earlier!! don't know which place is better
    # to ensure this
    # start = static.data[:, :, 0].unsqueeze(1)
    # y = torch.cat((start, tour, start), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(tour[:, :-1] - tour[:, 1:], 2), dim=2)) # [batch_size, sequence_length]

    return tour_len.sum(1)

# class Decoder(nn.Module):
#     def __init__(self, sequence_length):
#         super().__init__()
#         self.decoder = nn.RNN(input_size=hidden_size,
#                               hidden_size=hidden_size,
#                               batch_first=True)
#
#
#         self.decoderOutputToActualProbabilities = nn.Linear(hidden_size,sequence_length)
#
#         self.targetToDecoderInput = nn.Linear(1, hidden_size)
#     def apply_mask_to_logits(self, logits, mask, indexes):
#         batch_size = logits.size(0)
#         clone_mask = mask.clone()
#
#         if indexes is not None:
#             clone_mask[[i for i in range(batch_size)], indexes.data.squeeze(1).long()] = 1
#
#             logits[clone_mask.unsqueeze(1)] = -np.inf
#         else:
#             logits[:, :] = -np.inf
#             # we want to start from depot, ie the first node
#             logits[:,:,0] = 1
#         return logits, clone_mask
#
#     def forward(self, points, encoder_context, targets=None):
#         '''
#         input: [batch_size, num_nodes (sequence_length), static_features]
#         '''
#
#         batch_size = points.size(0)
#         time_steps = points.size(1)  # kanei tosa bhmata osa to sequence length poy dinetai
#
#         #decoder_input = torch.ones(batch_size, 1, hidden_size)
#         # trying to pass the first decoder input through the embedding
#         decoder_input = self.targetToDecoderInput(torch.ones(batch_size, 1)).unsqueeze(1)
#
#         # we calculate loss per batch here
#         loss = 0
#         tours = []
#         tour_logp = []
#
#         chosen_indexes = None
#         mask = torch.zeros(batch_size, time_steps).byte()
#
#         context = encoder_context
#         for time_step in range(time_steps):
#             assert decoder_input.size(0) == batch_size
#             assert decoder_input.size(1) == 1 # sequence size is 1, takes 1 by 1 the input seq
#             assert decoder_input.size(2) == hidden_size
#
#             decoder_output,  context = self.decoder(decoder_input, context)
#
#             # TODO: figure out pws pairnoume to actual result apo to decoder output?
#             logits = self.decoderOutputToActualProbabilities(decoder_output) # [batch_size, seq_len ]
#
#             ## TODO: add mask!
#             masked_logits, mask = self.apply_mask_to_logits(logits, mask, chosen_indexes)
#
#
#             softmax_logits = F.softmax(masked_logits, dim=2) # exei size [batch_size, 1]
#
#             isAllNan = torch.all(softmax_logits.isnan())
#             if isAllNan:
#                 break
#
#             # chosen_indexes = torch.argmax(softmax_logits, dim=2).float()
#
#             # m = torch.distributions.Categorical(softmax_logits) # edw xanetai to requires_grad
#             # ptr = m.sample() # [batch_size, 1]
#             # chosen_indexes = ptr.data.detach()  #
#             # logp = m.log_prob(ptr)
#
#             ### When we use argmax
#             chosen_indexes = torch.argmax(softmax_logits, dim=2).float() # [batch_size, 1]
#             log_probs = F.log_softmax(softmax_logits, dim=2)
#             logp = torch.gather(log_probs, 2, chosen_indexes.unsqueeze(2).long()).squeeze(2)
#
#             assert logp.requires_grad == True
#
#             tour_logp.append(logp.unsqueeze(1))
#             tours.append(chosen_indexes.unsqueeze(1))
#
#             # decoder input is next item in targets
#             #decoder_input = self.targetToDecoderInput(targets[:, time_step].unsqueeze(1).float()).unsqueeze(1)
#             decoder_input = self.targetToDecoderInput(chosen_indexes.float()).unsqueeze(1)
#
#         tours = torch.cat(tours, 2)
#         tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)
#
#         return   tours,tour_logp


class ClassicSeq2SeqTSPModel(nn.Module):
    def __init__(self,sequence_length):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(sequence_length)
    def forward(self, inputs, targets=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        tours, tour_logp = self.decoder(encoder_outputs, encoder_hidden, targets)

        return tours,tour_logp

def trainClassicSeq2SeqTSPWithReinforcementLearning(train_dataset,
                                                    test_dataset,
                                                    epochs,
                                                    experiment_details,
                                                    batch_size = 10,
                                                    num_nodes=13,
                                                    lr=1e-4):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    model = ClassicSeq2SeqTSPModel(sequence_length=num_nodes)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    val_loss = []
    train_loss= []
    val_loss_per_epoch = []
    losses_per_epoch = []

    tour_lengths_per_epoch = []

    for epoch in range(epochs):
        model.train()

        loss_at_epoch = 0.0
        val_loss_at_epoch = 0.0
        iterator = tqdm(train_loader, unit='Batch')
        average_tour_length_per_epoch = []

        for batch_id, sample_batch in enumerate(iterator):
            optimizer.zero_grad()

            train_batch = Variable(sample_batch['Points'])
            # target_batch = Variable(sample_batch['Solution'])

            #output_routes,tour_logp = model(train_batch, target_batch)
            output_routes, tour_logp = model(train_batch)

            reward = reward_fn(train_batch.transpose(1, 2), output_routes.squeeze(1).to(torch.int64))

           # ?? we use a negative because optimizers use gradient descent, whilst the rule above assumes gradient ascent.
           #  loss = torch.mean(reward.detach() * (tour_logp))
            # loss = torch.sum(reward * tour_logp.squeeze(2))
            #TODO: figure out what loss is, check this out too https://pytorch.org/docs/stable/distributions.html
            loss = torch.sum(reward * (-tour_logp))
            # look at the bug when you add the -!! you have to understand what the - is

            loss.backward()
            optimizer.step()
            loss_at_epoch += loss.detach().sum().item()
            distances_per_batch = reward_fn(sample_batch['Points'].transpose(1,2), output_routes.squeeze(1).to(torch.int64))
            average_tour_length_per_epoch.append(distances_per_batch.sum(0).item())

        tour_lengths_per_epoch.append(np.mean(average_tour_length_per_epoch))
        # epoch finished
        model.eval()
        for val_batch in validation_loader:
            train_batch = Variable(val_batch['Points'])
            target_batch = Variable(val_batch['Solution'])
            # tours, tour_logp = model(train_batch, target_batch)
            tours, tour_logp = model(train_batch )

            reward = reward_fn(train_batch.transpose(1, 2), tours.squeeze(1).to(torch.int64))

            # TODO: check this https://pytorch.org/docs/stable/distributions.html
            # loss = torch.mean(reward.detach() * (-tour_logp.sum(dim=1)))
            # loss = torch.mean(reward.detach() * (-tour_logp))
            val_loss_at_batch = torch.sum(reward * (-tour_logp.squeeze(2)))#torch.mean(reward.detach() * (tour_logp))

            val_loss.append(val_loss_at_batch.data.item())
            val_loss_at_epoch += val_loss_at_batch.detach().item()

            # for this tour, what did the model find and what did the target was?
            format_tours(target_batch, tours.squeeze(1).int())
            #format_tour(tours.squeeze(1).int())



        train_loss.append(loss_at_epoch/batch_size)

        print(f'Loss at epoch {epoch}: {loss_at_epoch}')
        losses_per_epoch.append(loss_at_epoch)
        val_loss_per_epoch.append(val_loss_at_epoch)

    # Training finished
    plot_train_and_validation_loss(epoch, losses_per_epoch, val_loss_per_epoch, experiment_details, "classicSeqToSeq")


    plot_average_tour_length(tour_lengths_per_epoch, experiment_details)

    return model,tours


if __name__ == '__main__':

    epochs = 10
    num_nodes = 5
    train_size = 500
    test_size = 25
    batch_size = 5
    lr = 1e-4

    train_dataset = TSPDataset(train_size, num_nodes)
    test_dataset = TSPDataset(test_size, num_nodes)
    # train_dataset = TSPDatasetWithoutSolutions(train_size, num_nodes)
    # test_dataset = TSPDatasetWithoutSolutions(test_size, num_nodes)

   # experiment_details = f'ARGMAX_REINFORCEMENT_epochs{epochs}_train{train_size}_seqLen{num_nodes}_batch{batch_size}_lr{lr}'
    experiment_details = f'NEW_MODEL_epochs{epochs}_train{train_size}_seqLen{num_nodes}_batch{batch_size}_lr{lr}'

    trainClassicSeq2SeqTSPWithReinforcementLearning(train_dataset,
                                        test_dataset,
                                        epochs,
                                        experiment_details,
                                        batch_size ,
                                        num_nodes,
                                        lr)