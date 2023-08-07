import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from datasets.DEPRECATED_CVRP_dataset import update_dynamic, should_terminate_cvrp
from models.Attention import Attention
from models.Embeddings.ConvolutionalEmbedding import ConvolutionalEncoder
from models.Embeddings.GraphEmbeddings.GraphAttentionEncoderCVRP import GraphAttentionEncoderForCVRP
from models.EncoderDecoder import Decoder

static_features = 2
dynamic_features = 2
hidden_size = 128
capacity = 60
velocity = 60
cons_rate =0.2 # fuel consumption rate (gallon/mile)
t_limit = 11
num_afs = 3

### MODEL CHOICE:
use_seperate_decoder_input_embedding = True
dropout_decoder_hidden_states = True

class MHA_CVRP_solver(nn.Module):
    def __init__(self):
        super().__init__()
        dropout= 0.5
        self.encoder_static = GraphAttentionEncoderForCVRP()
        self.encoder_dynamic = ConvolutionalEncoder(in_feats=dynamic_features, out_feats=hidden_size)
        self.drop_hh = nn.Dropout(p=dropout)
        self.embedding_for_decoder_input = ConvolutionalEncoder(in_feats=static_features, out_feats=hidden_size)

        self.decoder_for_one_seq_len = Decoder(in_feats=hidden_size,hidden_size=hidden_size)
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.attention = Attention(hidden_size)
        self.max_steps = 1000

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic, distances):
        '''
        static: [batch_size, static_features, seq_len],
        dynamic: [batch_size, dynamic_features, seq_len],
        distances: [bs, seq_len, sel_len] ?
        '''

        batch_size, _ ,seq_len = static.shape
        embeddings, mean_embeddings = self.encoder_static(static=static, dynamic=dynamic) # [batch_size, hidden_size, seq_len]
        embeddings = embeddings.transpose(2, 1)
        dynamic_embeddings = self.encoder_dynamic(dynamic)

        # mask: [batch_size, sequence_length]
        mask = self.mask.repeat(seq_len).unsqueeze(0).repeat(batch_size, 1)
        self.attention.init_inf(mask.size())


        if use_seperate_decoder_input_embedding:
            decoder_input = torch.ones(batch_size, static_features, 1)
            decoder_input = self.embedding_for_decoder_input(decoder_input)
        else:
            # always start from depot, ie the 1st point (0 index)
            decoder_input = embeddings[:,:, :1] # same as: static_embeddings[:,:,0].unsqueeze(2)

        decoder_input = decoder_input.transpose(2, 1)


        decoder_states = None

        outputs = []
        tours = []
        tour_logp= []

        old_idx = torch.zeros(batch_size, 1, dtype=torch.long)

        for i in range(self.max_steps):
            if should_terminate_cvrp(dynamic):  # all demands have been satisfied
                break

            assert decoder_input.size(0)== batch_size and decoder_input.size(1)== 1 and decoder_input.size(2) == hidden_size

            decoder_output, decoder_states = self.decoder_for_one_seq_len(decoder_input, decoder_states)

            decoder_hidden_state = decoder_states[0]

            assert embeddings.size(0) == batch_size and embeddings.size(1) == hidden_size and embeddings.size(2) == seq_len
            use_attention_aware_hidden_states= False
            if use_attention_aware_hidden_states:
                probability_to_visit_each_index,decoder_hidden_state = self.attention(decoder_hidden_state.squeeze(0),
                                                                                       embeddings,
                                                                                       dynamic_embeddings,
                                                                                       mask)
            else:
                probability_to_visit_each_index, _ = self.attention(decoder_hidden_state.squeeze(0),
                                                                embeddings,
                                                                dynamic_embeddings,
                                                                mask)

            assert decoder_hidden_state.size(0)== 1 and decoder_hidden_state.size(1)== batch_size and decoder_hidden_state.size(2) == hidden_size

            try:
                m = torch.distributions.Categorical(probability_to_visit_each_index)
                ptr = m.sample()
            except:
                print("ERROR")

            chosen_indexes = ptr.data.detach()
            logp = m.log_prob(ptr)
            tour_logp.append(logp.unsqueeze(1))

            assert old_idx.size(0) == batch_size and old_idx.size(1) == 1

            dynamic = update_dynamic(dynamic, chosen_indexes)

            dynamic_embeddings = self.encoder_dynamic(dynamic)

            old_idx = chosen_indexes.unsqueeze(1)

            outputs.append(probability_to_visit_each_index.unsqueeze(0))  # [batch_size, seq_len]-> [1, bs,seq_len]
            tours.append(chosen_indexes.unsqueeze(1))  # [batch_size] -> [bs,1]


            if use_seperate_decoder_input_embedding:
                decoder_input = static[torch.arange(batch_size),:, chosen_indexes].unsqueeze(1).transpose(2,1)
                decoder_input = self.embedding_for_decoder_input(decoder_input).transpose(2,1)
            else:
                # update decoder input to chosen indexes static embeddings
                decoder_input = embeddings[torch.arange(batch_size), :, chosen_indexes].unsqueeze(1)

            # we update decoder states
            if dropout_decoder_hidden_states:
                decoder_states = (self.drop_hh(decoder_hidden_state), self.drop_hh(decoder_states[1]))
            else:
                decoder_states = ( decoder_hidden_state ,  decoder_states[1] )


        outputs = torch.cat(outputs).permute(1, 0, 2)
        tours = torch.cat(tours, 1)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        assert outputs.size(0) == batch_size and outputs.size(2) == seq_len
        assert tours.size(0) == batch_size
        return tours, tour_logp  # outputs:[bs, seq_len, seq_len], tours: [bs, seq_len]
