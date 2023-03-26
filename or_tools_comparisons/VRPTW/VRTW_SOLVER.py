import torch
import torch.nn as nn

## TODO: create a model that can solve VRPTW
from datasets.VRPTW_dataset import update_dynamic_state_vrptw, update_mask_vrptw
from models.AttentionVariations.VRPTW_Attention import DoubleAttention

STATIC_SIZE = 4
DYNAMIC_SIZE = 1
HIDDEN_SIZE = 128

class Encoder(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.num_of_features = in_feats
        self.encoder = nn.Linear(in_features=in_feats, out_features=out_feats)

    def forward(self, input):
        '''
        input: [batch_size, num_nodes, num_features]
        '''
        assert input.size(2) == self.num_of_features
        return self.encoder(input)

class PointerNetworkVRPTW(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_static = Encoder(in_feats=STATIC_SIZE, out_feats= HIDDEN_SIZE)
        self.decoder = nn.GRU(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)

        self.dynamic_embedding = Encoder(in_feats=DYNAMIC_SIZE, out_feats= HIDDEN_SIZE)
        self.decoder_input_dynamic_embedding = Encoder(in_feats=HIDDEN_SIZE, out_feats=HIDDEN_SIZE)

        self.attention = DoubleAttention(HIDDEN_SIZE)
        dropout = 0.2
        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hidden = nn.Dropout(p=dropout)

        self.max_steps = 1000
    def forward(self, static, dynamic, distance_matrix):
        '''
         static: [batch_size, static_features, sequence_length]
         dynamic: [batch_size, dynamic_features, sequence_length]
        '''

        batch_size = static.size(0)
        seq_len = static.size(2)
        static_embeddings = self.encoder_static(static.transpose(1,2))

        # to input tou decoder einai ta static embeddings only features tou prwtou city
        decoder_input = static_embeddings[:,0:1, :]
        # ta perimenei se morfh: [batch_size, static_feats , seq_len=1]

        tours = []
        tour_logp= []


        mask = torch.ones(batch_size, seq_len)
        initial_mask = torch.ones(batch_size, seq_len)
        self.attention.init_inf(mask.size())

        dynamic_embeddings = self.dynamic_embedding(dynamic.transpose(1,2))

        # old indexes are initially the depot. it has size [batch_size] bc for each batch we chose the first
        # location ie the depot
        old_idx = torch.zeros(batch_size)
        decoder_states = None
        for i in range(self.max_steps):

            if not mask.byte().any():
                # an ola sto mask einai 0 aka den pas pouthena, terminate
                break

            decoder_input = self.decoder_input_dynamic_embedding(decoder_input)
            decoder_input = self.drop_rnn(decoder_input)

            assert decoder_input.size(0) == batch_size and decoder_input.size(1) == 1 and decoder_input.size(2) == HIDDEN_SIZE
            decoder_output, decoder_states = self.decoder(decoder_input, decoder_states)

            # apply dropout to decoder states
            decoder_states = self.drop_hidden(decoder_states)

            if i == 0:
                # we are at depot. We must start from depot
                initial_mask[:, 1:] = 0

                # True: einai masked. False:den einai masked
                # 0: den mporeis na pas, 1: mporeis na pas
                mask_with_booleans = torch.eq(initial_mask, 0)
            else:
                # 0 means we can't go -> transform to True as in True its masked
                mask_with_booleans = torch.eq(mask, 0)

            probability_to_visit_each_vertex = self.attention(static_embeddings.transpose(2,1),
                                                              dynamic_embeddings,
                                                              decoder_output.transpose(2,1),mask_with_booleans)

            m = torch.distributions.Categorical(probability_to_visit_each_vertex)
            ptr = m.sample()
            chosen_indexes = ptr.data.detach()
            logp = m.log_prob(ptr)


            tour_logp.append(logp.unsqueeze(1))
            tours.append(chosen_indexes.unsqueeze(1))

            # we update mask and dynamic representation mask:
            # update dynamic state + embeddings
            dynamic = update_dynamic_state_vrptw( static, dynamic, chosen_indexes, old_idx, distance_matrix )

            dynamic_embeddings = self.dynamic_embedding(dynamic.transpose(2,1))



            mask = update_mask_vrptw(static, dynamic, mask, chosen_indexes)
            old_idx = chosen_indexes

        time_spent_at_each_route = dynamic.clone().squeeze(1)[:, 0]

        tours = torch.cat(tours, 1)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tours, tour_logp, time_spent_at_each_route

class VRPTW_SOLVER_MODEL(nn.Module):
    def __init__(self):
        super().__init__()

        self.pntr_network_vrptw = PointerNetworkVRPTW()
    def forward(self, static, dynamic, distance_matrix):
         # xreiazomaste ton distance matrix giati gia na ananewsoume to mask
         # prepei na xeroume posh apostash dianhse -> posa metra xreiasthke apo thn mia
         # topothesia sthn allh
         return self.pntr_network_vrptw(static, dynamic, distance_matrix)