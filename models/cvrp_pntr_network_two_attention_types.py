import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# source: https://github.com/higgsfield/np-hard-deep-reinforcement-learning/blob/master/Neural%20Combinatorial%20Optimization.ipynb

class Attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, C=10, name='Bahdanau'):
        super(Attention, self).__init__()

        self.use_tanh = use_tanh
        self.C = C
        self.name = name

        if name == 'Bahdanau':
            self.W_query = nn.Linear(hidden_size, hidden_size)
            self.W_ref = nn.Conv1d(hidden_size, hidden_size, 1, 1)

            V = torch.FloatTensor(hidden_size)

            self.V = nn.Parameter(V)
            self.V.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

    def forward(self, query, ref):
        """
        Args:
            query: [batch_size x hidden_size]
            ref:   ]batch_size x seq_len x hidden_size]
        """

        batch_size = ref.size(0)
        seq_len = ref.size(1)

        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
            ref = self.W_ref(ref)  # [batch_size x hidden_size x seq_len]
            expanded_query = query.repeat(1, 1, seq_len)  # [batch_size x hidden_size x seq_len]
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size x 1 x hidden_size]
            logits = torch.bmm(V, F.tanh(expanded_query + ref)).squeeze(1)

        elif self.name == 'Dot':
            query = query.unsqueeze(2)

            # mallon ta logits einai to probability to visit each index
            logits = torch.bmm(ref, query).squeeze(2)  # [batch_size x seq_len x 1]
            ref = ref.permute(0, 2, 1)

        else:
            raise NotImplementedError

        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        else:
            logits = logits

        return ref, logits

# TODO: use this for embedding
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

class PointerNet(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh):
        super(PointerNet, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len

        self.embedding = nn.Embedding(seq_len, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration )
        self.glimpse = Attention(hidden_size, use_tanh=False )

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

        # self.criterion = nn.CrossEntropyLoss()

    def apply_mask_to_logits(self, logits, mask, idxs):

        # TODO: fix this to use our mask
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf
        return logits, clone_mask

    def forward(self, inputs):
        #TODO: OMG ADD DYNAMIC DYMENSION!!
        """ Args:  inputs: [batch_size x sourceL]  """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        assert seq_len == self.seq_len

        embedded = self.embedding(inputs)


        encoder_outputs, (hidden, context) = self.encoder(embedded)

        mask = torch.zeros(batch_size, seq_len).byte()

        idxs = None

        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        loss = 0

        tours = []
        tour_logp = []

        for i in range(seq_len):

            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            query = hidden.squeeze(0)

            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                query = torch.bmm(ref, F.softmax(logits).unsqueeze(2)).squeeze(2)

            _, logits = self.pointer(query, encoder_outputs)

            # probably we should use these logits to chose indexes
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)

            decoder_input = inputs[:, i, :] #target_embedded[:, i, :]

            # this is for supervised learning. we want rl
            # loss += self.criterion(logits, target[:, i])


            # based on the logits we have, we calculate the posibility to
            # visit each index
            probability_to_visit_each_vertex = F.softmax(logits, dim=1)  # (batch, seq_len)
            m = torch.distributions.Categorical(probability_to_visit_each_vertex)
            ptr = m.sample()
            chosen_indexes = ptr.data.detach()
            logp = m.log_prob(ptr)

            # we store the result
            tour_logp.append(logp.unsqueeze(1))
            tours.append(chosen_indexes.unsqueeze(1))

        #they were returning: return loss / seq_len

        return tours, tour_logp