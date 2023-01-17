import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from EVRP.dataset import update_fn
from EVRP.models.Attention import Attention
from EVRP.models.EncoderDecoder import Encoder,Decoder

static_features = 2
dynamic_features = 3
hidden_size = 128
capacity = 60
velocity = 60
cons_rate =0.2 # fuel consumption rate (gallon/mile)
t_limit = 11
num_afs = 3

class EVRP_Solver(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_static = Encoder(in_feats=static_features, out_feats=hidden_size)
        self.encoder_dynamic = Encoder(in_feats=dynamic_features, out_feats=hidden_size)

        self.decoder_for_one_seq_len = Decoder(in_feats=hidden_size,hidden_size=hidden_size)
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.attention = Attention(hidden_size)
        self.max_steps = 1000
    def forward(self, static, dynamic,distances):
        '''
        static: [batch_size, static_features, seq_len],
        dynamic: [batch_size, dynamic_features, seq_len],
        distances: [bs, seq_len, sel_len] ?
        '''

        batch_size, hid,seq_len = static.shape
        static_embeddings = self.encoder_static(static)
        dynamic_embeddings = self.encoder_dynamic(dynamic)

        # mask: [batch_size, sequence_length]
        mask = self.mask.repeat(seq_len).unsqueeze(0).repeat(batch_size, 1)
        self.attention.init_inf(mask.size())

        # always start from depot, ie the 1st point. TODO: gt pairnei :1 kai oxi 1 sketo?
        decoder_input = static_embeddings[:,:, :1] # same as: static_embeddings[:,:,0].unsqueeze(2)
        decoder_input = decoder_input.transpose(2, 1)
        decoder_states = None

        outputs = []
        tours = []
        tour_logp= []

        old_idx = torch.zeros(batch_size, 1, dtype=torch.long)
        dis_by_afs = [distances[:, i:i + 1, 0:1] + distances[:, i:i + 1, :] for i in range(1, num_afs + 1)]
        dis_by_afs = torch.cat(dis_by_afs, dim=1)
        dis_by_afs = torch.min(dis_by_afs, dim=1)  # tuple: (batch, seq_len), ()
        dis_by_afs[0][:, 0] = 0

        for i in range(self.max_steps):
            if (dynamic[:, 2, :] == 0).all():  # all demands have been satisfied
                break

            assert decoder_input.size(0)== batch_size and decoder_input.size(1)== 1 and decoder_input.size(2) == hidden_size

            decoder_output, decoder_states = self.decoder_for_one_seq_len(decoder_input, decoder_states)

            decoder_hidden_state = decoder_states[0]

            probability_to_visit_each_index,decoder_hidden_state = self.attention(decoder_hidden_state.squeeze(0),
                                                                                   static_embeddings,
                                                                                   dynamic_embeddings,
                                                                                   mask)

            assert decoder_hidden_state.size(0)== 1 and decoder_hidden_state.size(1)== batch_size and decoder_hidden_state.size(2) == hidden_size

            m = torch.distributions.Categorical(probability_to_visit_each_index)
            ptr = m.sample()
            chosen_indexes = ptr.data.detach()
            logp = m.log_prob(ptr)
            tour_logp.append(logp.unsqueeze(1))

            assert old_idx.size(0) == batch_size and old_idx.size(1) == 1
            dynamic = update_fn(old_idx, chosen_indexes.unsqueeze(1),
                                     mask, dynamic,
                                     distances, dis_by_afs, capacity,
                                     velocity, cons_rate, t_limit,
                                     num_afs)  # update mask and dynamic
            dynamic_embeddings = self.encoder_dynamic(dynamic)

            old_idx = chosen_indexes.unsqueeze(1)

            outputs.append(probability_to_visit_each_index.unsqueeze(0))  # [batch_size, seq_len]-> [1, bs,seq_len]
            tours.append(chosen_indexes.unsqueeze(1))  # [batch_size] -> [bs,1]

            # update decoder input to chosen indexes static embeddings
            decoder_input = static_embeddings[torch.arange(batch_size), :, chosen_indexes].unsqueeze(1)

            # we update decoder states
            decoder_states = (decoder_hidden_state, decoder_states[1])


        outputs = torch.cat(outputs).permute(1, 0, 2)
        tours = torch.cat(tours, 1)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        assert outputs.size(0) == batch_size and outputs.size(2) == seq_len
        assert tours.size(0) == batch_size
        return outputs, tours, tour_logp  # outputs:[bs, seq_len, seq_len], tours: [bs, seq_len]
