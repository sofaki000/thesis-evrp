import torch
import torch.nn as nn
from CVRP_Attention import Attention
from datasets.capacitated_vrp_dataset import update_dynamic, update_mask

static_features = 2
dynamic_features = 2
hidden_size = 128

class Embedding(nn.Module):
    def __init__(self,input_feats, out_feats):
        super().__init__()
        self.embedding = nn.Conv1d(in_channels=input_feats, out_channels=out_feats,kernel_size=1)

    def forward(self, input):
        '''
            input: [batch_size, features, seq_len]
        '''
        return self.embedding(input)

class PointerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.dynamic_embedding = Embedding(input_feats=dynamic_features, out_feats=hidden_size)

        self.decoder_input_dynamic_embedding = Embedding(input_feats=hidden_size, out_feats=hidden_size)

        self.attention = Attention(hidden_size)

        self.max_steps=1000
    def initialize_decoder_states(self, batch_size):
        hidden = torch.zeros(1,batch_size,hidden_size)
        return (hidden, hidden)
    def forward(self, static, static_embeddings, dynamic):
        '''
        static: [batch_size, static_features, sequence_length]
        dynamic: [batch_size, dynamic_features, sequence_length]
        static_embeddings:[]
        '''
        batch_size = static.size(0)
        seq_len = static.size(2)

        hidden, context =  self.initialize_decoder_states(batch_size)

        # to input tou decoder einai ta static embeddings only features.
        # ta perimenei se morfh: [batch_size, static_feats , seq_len=1]
        decoder_input = static_embeddings[:, :, 0:1].transpose(2,1)

        tours = []
        tour_logp= []

        mask = torch.ones(batch_size, seq_len)
        self.attention.init_inf(mask.size())

        dynamic_embeddings = self.dynamic_embedding(dynamic)

        for i in range(self.max_steps):

            if not mask.byte().any():
                # an ola sto mask einai 0 aka den pas pouthena, terminate
                break

            decoder_input = self.decoder_input_dynamic_embedding(decoder_input.transpose(2,1)).transpose(2,1)
            assert decoder_input.size(0) == batch_size and decoder_input.size(1) == 1 and decoder_input.size(2) == hidden_size
            decoder_output, (hidden, context) = self.decoder(decoder_input, (hidden, context))

            if i == 0:
                # we are at depot. We must start from depot
                mask[:, 1:] = 0
            mask_with_booleans = torch.eq(mask, 0)
            probability_to_visit_each_vertex = self.attention(static_embeddings, dynamic_embeddings, decoder_output,mask_with_booleans)
            m = torch.distributions.Categorical(probability_to_visit_each_vertex)
            ptr = m.sample()
            chosen_indexes = ptr.data.detach()
            logp = m.log_prob(ptr)

            tour_logp.append(logp.unsqueeze(1))
            tours.append(chosen_indexes.unsqueeze(1))

            # we update mask and dynamic representation mask:
            # update dynamic state + embeddings
            dynamic = update_dynamic(dynamic, chosen_indexes)
            dynamic_embeddings = self.dynamic_embedding(dynamic)

            mask = update_mask(mask, dynamic, chosen_indexes)

        tours = torch.cat(tours, 1)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)
        return tours, tour_logp

class CVRPSolver(nn.Module):
    def __init__(self):
        super().__init__()

        self.static_embedding = Embedding(input_feats=static_features, out_feats=hidden_size)
        self.decoder = PointerNetwork()

    def forward(self, static, dynamic):
        '''static: [batch_size, static_features, seq_len],dynamic: [batch_size, dynamic_features, seq_len]'''
        static_embeddings = self.static_embedding(static)

        tours, tour_logp = self.decoder(static, static_embeddings, dynamic)

        return tours, tour_logp


