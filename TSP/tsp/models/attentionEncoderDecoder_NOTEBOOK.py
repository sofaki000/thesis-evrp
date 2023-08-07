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

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class Decoder(nn.Module):
    def __init__(self, sequence_length):
        super(Decoder, self).__init__()
        dropout_p = 0.1
        self.embedding = nn.Linear(1, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, sequence_length)
        self.dropout = nn.Dropout(dropout_p)

    def apply_mask_to_logits(self, logits, mask, indexes):
        batch_size = logits.size(0)
        clone_mask = mask.clone()
        if indexes is not None:
            clone_mask[[i for i in range(batch_size)], indexes.data.squeeze(1).long()] = 1

            logits[clone_mask.unsqueeze(1)] = -np.inf
        else:
            logits[:, :] = -np.inf
            # we want to start from depot, ie the first node
            logits[:, :, 0] = 1

        return logits, clone_mask

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        sequence_length = encoder_outputs.size(1)
        decoder_input = torch.ones(batch_size, 1)
        decoder_hidden = encoder_hidden

        attentions = []
        tours = []
        tour_logp = []
        mask = torch.zeros(batch_size, sequence_length).byte()

        chosen_indexes = None
        for i in range(sequence_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step_decoder(decoder_input,
                                                                             decoder_hidden,
                                                                             encoder_outputs)

            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1).float() # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach().float()  # detach from history as input

            masked_logits, mask = self.apply_mask_to_logits(decoder_output, mask, chosen_indexes)

            # We transform decoder output to the actual result
            chosen_indexes = torch.argmax(masked_logits, dim=2).float()  # [batch_size, 1]
            log_probs = F.log_softmax(decoder_output, dim=2)
            logp = torch.gather(log_probs, 2, chosen_indexes.unsqueeze(2).long()).squeeze(2)  # [batch_size, 1]

            tour_logp.append(logp.unsqueeze(1))
            tours.append(chosen_indexes.unsqueeze(1))

        attentions = torch.cat(attentions, dim=1)
        tours = torch.cat(tours, 2)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tours, tour_logp, attentions


    def forward_step_decoder(self, decoder_input, decoder_hidden, encoder_outputs):

        embedded =  self.dropout(self.embedding(decoder_input)).unsqueeze(1)

        query = decoder_hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, decoder_hidden = self.gru(input_gru, decoder_hidden)
        output = self.out(output)

        return output, decoder_hidden, attn_weights

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
    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(tour[:, :-1] - tour[:, 1:], 2), dim=2)) # [batch_size, sequence_length]

    return tour_len.sum(1)



class ClassicSeq2SeqTSPModel(nn.Module):
    def __init__(self,sequence_length):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(sequence_length)
    def forward(self, inputs, targets=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        tours, tour_logp, attentions = self.decoder(encoder_outputs, encoder_hidden, targets)
        return tours,tour_logp


def train_epoch(model, dataloader,  optimizer):

    total_loss = 0
    distancePerEpoch = 0

    for data in dataloader:

        input_tensor = Variable(data['Points'])
        target_tensor = Variable(data['Solution'])

        optimizer.zero_grad()

        output_routes, tour_logp = model(input_tensor, target_tensor)

        reward = reward_fn(input_tensor.transpose(1, 2),
                           output_routes.squeeze(1).to(torch.int64))

        loss = torch.sum(reward * (tour_logp))

        loss.backward()

        optimizer.step()
        distancePerEpoch += torch.sum(reward.detach())

        total_loss += loss.item()

    return total_loss / len(dataloader), distancePerEpoch

def train(train_dataloader, model, n_epochs, learning_rate=0.001, print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    tour_lengths_per_epoch = []

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, n_epochs + 1):
        loss, distancePerEpoch = train_epoch(model, train_dataloader, optimizer)
        print_loss_total += loss
        plot_loss_total += loss

        tour_lengths_per_epoch.append(distancePerEpoch)

        if True: #epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if True: #epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # Training finished
    plot_train_and_validation_loss(epoch, plot_losses, plot_losses, experiment_details, "attentionModelLosses")
    plot_average_tour_length(tour_lengths_per_epoch, experiment_details)


import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np



if __name__ == '__main__':

    epochs = 100
    num_nodes = 5
    train_size = 1000
    test_size = 100
    batch_size = 5
    lr = 1e-4

    train_dataset = TSPDataset(train_size, num_nodes)
    test_dataset = TSPDataset(test_size, num_nodes)

    experiment_details = f'ATTENTION_DETE_epochs{epochs}_train{train_size}_seqLen{num_nodes}_batch{batch_size}_lr{lr}'

    model = ClassicSeq2SeqTSPModel(sequence_length=num_nodes)
    train_dataloader = DataLoader(train_dataset,   batch_size=batch_size)
    train(train_dataloader, model, epochs, learning_rate=0.001, print_every=100, plot_every=100)
    # trainClassicSeq2SeqTSPWithReinforcementLearning(train_dataset,
    #                                     test_dataset,
    #                                     epochs,
    #                                     experiment_details,
    #                                     batch_size ,
    #                                     num_nodes,
    #                                     lr)