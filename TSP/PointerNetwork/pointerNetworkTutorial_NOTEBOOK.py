import datetime
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


from IPython.display import clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt

from ploting.plot_utilities import get_filename_time, plot_train_and_validation_loss, plot_train_tour_length


class TSPDataset(Dataset):
    def __init__(self, num_nodes, num_samples, random_seed=111):
        super(TSPDataset, self).__init__()
        torch.manual_seed(random_seed)

        self.data_set = []
        for l in tqdm(range(num_samples)):
            x = torch.FloatTensor(2, num_nodes).uniform_(0, 1)
            self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]



def reward(sample_solution ):
    """
    Args:
        sample_solution seq_len of [batch_size]
    """
    batch_size = sample_solution[0].size(0)
    n = len(sample_solution)
    tour_len = Variable(torch.zeros([batch_size]))


    for i in range(n - 1):
        tour_len += torch.norm(sample_solution[i] - sample_solution[i + 1], dim=1)

    tour_len += torch.norm(sample_solution[n - 1] - sample_solution[0], dim=1)

    return tour_len


class Attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, C=10, name='Bahdanau' ):
        super(Attention, self).__init__()

        self.use_tanh = use_tanh
        self.C = C
        self.name = name

    def forward(self, query, ref):
        """
        Args:
            query: [batch_size x hidden_size]
            ref:   ]batch_size x seq_len x hidden_size]
        """

        batch_size = ref.size(0)
        seq_len = ref.size(1)
        query = query.unsqueeze(2)
        logits = torch.bmm(ref, query).squeeze(2)  # [batch_size x seq_len x 1]
        ref = ref.permute(0, 2, 1)


        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        else:
            logits = logits
        return ref, logits





class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size ):
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

class PointerNet(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 attention ):
        super(PointerNet, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len

        self.embedding = GraphEmbedding(2, embedding_size )
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration, name=attention)
        self.glimpse = Attention(hidden_size, use_tanh=False, name=attention )

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def apply_mask_to_logits(self, logits, mask, idxs):
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf
        return logits, clone_mask

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size x 1 x sourceL]
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(2)
        assert seq_len == self.seq_len

        embedded = self.embedding(inputs)
        encoder_outputs, (hidden, context) = self.encoder(embedded)

        prev_probs = []
        prev_idxs = []
        mask = torch.zeros(batch_size, seq_len).byte()


        idxs = None

        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        for i in range(seq_len):

            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            query = hidden.squeeze(0)
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                query = torch.bmm(ref, F.softmax(logits).unsqueeze(2)).squeeze(2)

            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            probs = F.softmax(logits)

            idxs = probs.multinomial(1).squeeze(1)
            for old_idxs in prev_idxs:
                if old_idxs.eq(idxs).data.any():
                    print
                    seq_len
                    print(' RESAMPLE!')
                    idxs = probs.multinomial().squeeze(1)
                    break
            decoder_input = embedded[[i for i in range(batch_size)], idxs.data, :]

            prev_probs.append(probs)
            prev_idxs.append(idxs)

        return prev_probs, prev_idxs


class CombinatorialRL(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 reward,
                 attention ):
        super(CombinatorialRL, self).__init__()
        self.reward = reward

        self.actor = PointerNet(
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            attention)

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, input_size, seq_len]
        """
        batch_size = inputs.size(0)
        input_size = inputs.size(1)
        seq_len = inputs.size(2)

        probs, action_idxs = self.actor(inputs)

        actions = []
        inputs = inputs.transpose(1, 2)
        for action_id in action_idxs:
            actions.append(inputs[[x for x in range(batch_size)], action_id.data, :])

        action_probs = []
        for prob, action_id in zip(probs, action_idxs):
            action_probs.append(prob[[x for x in range(batch_size)], action_id.data])

        R = self.reward(actions )

        return R, action_probs, actions, action_idxs



class TrainModel:
    def __init__(self, model, train_dataset, val_dataset, seq_len, batch_size=128, threshold=None, max_grad_norm=2.):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.threshold = threshold
        self.seq_len = seq_len

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        self.actor_optim = optim.Adam(model.actor.parameters(), lr=1e-4)
        self.max_grad_norm = max_grad_norm

        self.train_tour = []
        self.val_tour = []

        self.epochs = 0

    def train_and_validate(self, n_epochs):
        critic_exp_mvg_avg = torch.zeros(1)

        losses_per_epoch = []
        for epoch in range(n_epochs):

            loss_at_epoch = 0.0
            for batch_id, sample_batch in enumerate(self.train_loader):
                self.model.train()

                inputs = Variable(sample_batch)

                R, probs, actions, actions_idxs = self.model(inputs)

                if batch_id == 0:
                    critic_exp_mvg_avg = R.mean()
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())

                advantage = R - critic_exp_mvg_avg

                logprobs = 0
                for prob in probs:
                    logprob = torch.log(prob)
                    logprobs += logprob
                logprobs[logprobs < -1000] = 0.

                reinforce = advantage * logprobs
                actor_loss = reinforce.mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.actor.parameters(),
                                              float(self.max_grad_norm),
                                              norm_type=2)

                self.actor_optim.step()

                critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

                self.train_tour.append(R.mean().data.item())

                loss_at_epoch += actor_loss.detach().item()

                if batch_id % 10 ==0:
                    self.plot(self.epochs)

                if batch_id % 10 == 0:

                    self.model.eval()
                    for val_batch in self.val_loader:
                        inputs = Variable(val_batch)

                        R, probs, actions, actions_idxs = self.model(inputs)
                        self.val_tour.append(R.mean().data.item())


            # epoch finished
            losses_per_epoch.append(loss_at_epoch)
            # if self.threshold and self.train_tour[-1] < self.threshold:
            #     print
            #     "EARLY STOPPAGE!"
            #     break

            self.epochs += 1


        # training finished
        experiment_details = f'epochs{self.epochs}_train{train_size}_batch{self.batch_size}'
        plot_train_and_validation_loss(epoch, losses_per_epoch, losses_per_epoch,experiment_details,"pntr")



    def plot(self, epoch):
        plot_train_tour_length(self.seq_len,
                               epoch,
                               self.train_tour,
                               self.val_tour,
                               train_size,
                               self.batch_size,
                               self.epochs)


if __name__ == '__main__':
    embedding_size = 128
    hidden_size = 128
    n_glimpses = 1
    tanh_exploration = 10
    use_tanh = True
    n_epochs = 45
    beta = 0.9
    max_grad_norm = 2.
    seq_len = 5

    tsp_20_model = CombinatorialRL(
        embedding_size,
        hidden_size,
        seq_len,
        n_glimpses,
        tanh_exploration,
        use_tanh,
        reward,
        attention="Dot")

    train_size = 100000#00  # 0000
    val_size = 10000#0  # 000
    train_20_dataset = TSPDataset(seq_len, train_size)
    val_20_dataset = TSPDataset(seq_len, val_size)


    tsp_20_train = TrainModel(tsp_20_model,
                            train_20_dataset,
                            val_20_dataset,
                              seq_len= seq_len,
                            threshold=3.99)
    tsp_20_train.train_and_validate(n_epochs)