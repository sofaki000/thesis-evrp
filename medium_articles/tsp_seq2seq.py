import math

import numpy as np
from scipy.spatial import distance_matrix
import torch.nn.functional as F
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from or_tools_comparisons.common_utilities import get_tour_length_from_distance_matrix
from ploting.metrics.plot_average_tour_length import plot_average_tour_length
from ploting.plot_utilities import plot_train_and_validation_loss
from torch.utils.data import Dataset

static_features = 2
hidden_size = 128

# input: [ [23,24], [34, 67], [34,56] ], current position: 2
# output: 0 -> 2 -> 1
# f(environment, input) = action = ax+bx^2 +

class TSPDataset(Dataset):
    """  Random TSP dataset """
    def __init__(self, data_size, seq_len):
        self.data_size = data_size
        self.seq_len = seq_len
        self.data = self._generate_data()
    def __len__(self):
        return self.data_size
    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data['Points_List'][idx]).float()
        sample = {'Points':tensor }
        return sample

    def _generate_data(self):
        """
        :return: Set of points_list ans their One-Hot vector solutions
        """
        points_list = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description('Data points %i/%i' % (i+1, self.data_size))
            points_list.append(np.random.randint(30,size=(self.seq_len, 2))) # np.random.random((self.seq_len, 2))

        return {'Points_List':points_list }

    def _to1hotvec(self, points):
        """
        :param points: List of integers representing the points indexes
        :return: Matrix of One-Hot vectors
        """
        vec = np.zeros((len(points), self.seq_len))
        for i, v in enumerate(vec):
            v[points[i]] = 1
        return vec

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        dropout_p = 0.1
        self.hidden_size = hidden_size
        self.embedding = GraphEmbedding(static_features, hidden_size)
        #self.embedding = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        '''
        input: The coordinates of the cities. [batch_size, sequence_length, 2]
        '''
        embedded = self.dropout(self.embedding(input.transpose(1,2)))
        output, hidden = self.gru(embedded)
        return output, hidden


class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(GraphEmbedding, self).__init__()
        self.embedding_size = embedding_size

        self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size))
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(2)
        embedding = self.embedding.repeat(batch_size, 1, 1)
        embedded = []
        inputs = inputs.unsqueeze(1)
        for i in range(seq_len):
            embedded.append(torch.bmm(inputs[:, :, :, i].float(), embedding))
        embedded = torch.cat(embedded, 1)
        return embedded

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
            logits[:, :, 0] = 1
        return logits, clone_mask

    def forward(self, encoder_outputs, encoder_hidden):
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

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

            # We use its own predictions as the next input
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(-1).detach().float()
            masked_logits, mask = self.apply_mask_to_logits(decoder_output, mask, chosen_indexes)

            # We transform decoder output to the actual result
            chosen_indexes = torch.argmax(masked_logits, dim=2).float()  # [batch_size, 1]
            log_probs = F.log_softmax(decoder_output, dim=2)
            logp = torch.gather(log_probs, 2, chosen_indexes.unsqueeze(2).long()).squeeze(2)  # batch_size, 1

            tour_logp.append(logp.unsqueeze(1))
            tours.append(chosen_indexes.unsqueeze(1))

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
    tour_len = torch.sqrt(torch.sum(torch.pow(tour[:, :-1] - tour[:, 1:], 2), dim=2))

    return tour_len.sum(1)


class ClassicSeq2SeqTSPModel(nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(sequence_length)

    def forward(self, inputs):
        encoder_outputs, encoder_hidden = self.encoder(inputs)

        tours, tour_logp = self.decoder(encoder_outputs, encoder_hidden)

        return tours, tour_logp


def trainClassicSeq2SeqTSPWithReinforcementLearning(train_dataset,
                                                    test_dataset,
                                                    epochs,
                                                    experiment_details,
                                                    batch_size=10,
                                                    num_nodes=13,
                                                    lr=1e-4):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    model = ClassicSeq2SeqTSPModel(sequence_length=num_nodes)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    val_loss = []
    train_loss = []
    val_loss_per_epoch = []
    losses_per_epoch = []

    tour_lengths_per_epoch = []

    for epoch in range(epochs):
        loss_at_epoch = 0.0
        model.train()
        val_loss_at_epoch = 0.0
        iterator = tqdm(train_loader, unit='Batch')

        for batch_id, sample_batch in enumerate(iterator):
            optimizer.zero_grad()
            train_batch = Variable(sample_batch['Points'])
            output_routes, tour_logp = model(train_batch)
            reward = reward_fn(train_batch.transpose(1, 2), output_routes.squeeze(1).to(torch.int64))

            # reward here is the distance travelled for the solution the model provided.
            # we want to minimize this distance and therefore we are using the log probabilities (which is the decoder
            # output) that contain a minus sign and we multiply it by the reward. We are using the REINFORCE algorithm
            loss = torch.mean(reward.detach() * tour_logp.sum(dim=1))

            loss.backward()
            optimizer.step()
            loss_at_epoch += loss.detach().sum().item()
            average_tour_length = 0

            for tour in range(batch_size):
                points = sample_batch['Points'][tour]
                distance_matrix_array = distance_matrix(points, points)
                content_from_my_model, tour_length = get_tour_length_from_distance_matrix(output_routes[tour].squeeze(0).long(), distance_matrix_array)
                average_tour_length += tour_length.item()

        model.eval()
        for val_batch in validation_loader:
            train_batch = Variable(val_batch['Points'])
            tours, tour_logp = model(train_batch)

            reward = reward_fn(train_batch.transpose(1, 2), tours.squeeze(1).to(torch.int64))

            loss = torch.mean(reward.detach() * tour_logp.sum(dim=1))

            val_loss.append(loss.data.item())
            val_loss_at_epoch += loss.detach().item()

        tour_lengths_per_epoch.append(average_tour_length / batch_size)
        train_loss.append(loss_at_epoch / batch_size)

        losses_per_epoch.append(loss_at_epoch)
        val_loss_per_epoch.append(val_loss_at_epoch)

    # Training finished
    plot_train_and_validation_loss(epoch, losses_per_epoch, val_loss_per_epoch, experiment_details, "classicSeqToSeq")
    plot_average_tour_length(tour_lengths_per_epoch, experiment_details)

    return model, tours


if __name__ == '__main__':
    epochs = 100
    num_nodes = 5
    train_size = 100000
    test_size = 100
    batch_size = 25
    lr = 1e-4
    train_dataset = TSPDataset(train_size, num_nodes)
    test_dataset = TSPDataset(test_size, num_nodes)

    experiment_details = f'GRAPH_EMBED_seq2seq_epochs{epochs}_train{train_size}_seqLen{num_nodes}_batch{batch_size}_lr{lr}'

    trainClassicSeq2SeqTSPWithReinforcementLearning(train_dataset,
                                                    test_dataset,
                                                    epochs,
                                                    experiment_details,
                                                    batch_size,
                                                    num_nodes,
                                                    lr)