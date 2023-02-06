
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

num_afs = 3
# attention structure
use_tahn = True


class Attention(nn.Module):
    def __init__(self, hidden_size, C=10):
        super().__init__()

        # self.w1 = nn.Linear(hidden_size, hidden_size)
        # self.w2_for_encoder = nn.Conv1d(hidden_size, hidden_size, 1, 1)
        # self.u = Parameter(torch.FloatTensor(hidden_size), requires_grad=True)
        # nn.init.uniform_(self.u, -1, 1)
        #
        # # auto to layer einai afou kanoume concatenate to attention aware hidden state
        # # kai to prohgoumeno tou decoder hidden state, na ta epanaferoume sto swsto dimension
        # self.hidden_out = nn.Linear(hidden_size * 2, hidden_size)
        self.V0 = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        self.V1 = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        self.V2 = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        self.exploring_c = 10
        self.context_linear1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.context_linear2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.linear = nn.Linear(hidden_size, hidden_size)

        nn.init.xavier_uniform_(self.V0)
        nn.init.xavier_uniform_(self.V1)
        nn.init.xavier_uniform_(self.V2)

        self.tanh = nn.Tanh()
        self._inf = nn.Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.softmax = nn.Softmax()
        self.linear_for_dec_hidden= nn.Linear(hidden_size*2, hidden_size)
        self.C = C
    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)
    def forward(self, decoder_hidden_state, static_embeddings, dynamic_embeddings, mask):
        '''
        :param decoder_hidden_state: [batch_size, hidden_size] QUERY
        :param static_embeddings: [batch_size, hidden_size, seq_len] REF
        :param dynamic_embeddings:  dynamic embedded inputs, [batch_size, hidden_size, seq_len]
        :mask: [batch_size, seq_len]: 0 if u can go, -inf if u can't go there
        :return:
        probs: probability to visit each vertex at next time step [batch_size, seq_len]
        new_decoder_hidden: attention aware decoder hidden state: [1, batch_size, hidden_size]
        '''
        batch_size = static_embeddings.size(0)
        seq_len = static_embeddings.size(2)

        v0 = self.V0.expand(batch_size, -1).unsqueeze(1)  # (batch, 1, hidden_size)
        v1 = self.V1.expand(batch_size, -1).unsqueeze(1)  # (batch, 1, hidden_size)
        v2 = self.V2.expand(batch_size, -1).unsqueeze(1)  # (batch, 1, hidden_size)
        e = self.context_linear1(static_embeddings)  # (batch, hidden_size, seq_len)
        e_plus = self.linear(decoder_hidden_state).unsqueeze(2).expand_as(e)
        e_dyna = self.context_linear2(dynamic_embeddings)
        e_sum = e + e_dyna + e_plus

        logits = torch.cat((v0.bmm(self.tanh(e_sum)[:, :, 0:1]),
                            v1.bmm(self.tanh(e_sum)[:, :, 1:num_afs + 1]),
                            v2.bmm(self.tanh(e_sum)[:, :, num_afs + 1:])), dim=2)

        if use_tahn:
            prob_to_visit_each_index = self.C * self.tanh(logits).squeeze(1)
        else:
            prob_to_visit_each_index = logits.squeeze(1)

        # exei 0 an den einai masked, -inf an einai masked to element dne mporoume na pame
        # ara den pairnei kamia pithanothta sto softmax
        probability_to_visit_each_index_masked = self.softmax(mask + prob_to_visit_each_index)

        probability_to_visit_each_index_masked = probability_to_visit_each_index_masked.squeeze(1)

        # we calculate context so we can concatenate decoder hidden states and create
        # the new decoder hidden states
        context = static_embeddings.bmm(probability_to_visit_each_index_masked.unsqueeze(2))  # [bs, hidden_size, 1]

        concatenated_states = torch.cat((decoder_hidden_state.unsqueeze(2), context), dim=1)


        attention_aware_hidden_state = self.C * self.tanh(self.linear_for_dec_hidden(concatenated_states.squeeze(2)).unsqueeze(0))



        return probability_to_visit_each_index_masked,attention_aware_hidden_state
        # w1_epi_encoder_states = self.w1(decoder_hidden_state).unsqueeze(2).expand(-1, -1, seq_len) # becomes: [bs, hidden_sz, seq_len]
        # w2_epi_decoder_states = self.w2_for_encoder(static_embeddings)
        #
        # # tanh_output: [batch_size, hidden_size, seq_len]
        # tanh_output = self.tanh(w1_epi_encoder_states + w2_epi_decoder_states)
        #
        # # v: [batch_size, 1, hidden_size]
        # v = self.u.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
        #
        # u_t = torch.bmm(v, tanh_output).squeeze(1)  # result:[batch_size,seq_len]
        #
        # if len(u_t[mask_with_booleans]) > 0:
        #     # to self.inf exei apla float('-inf') se bolikh morfh. bazeis kapoia inf wste sto epomeno
        #     # bhma sthn softmax na mhn exoun kanena chance na dialextoun
        #     u_t[mask_with_booleans] = self.inf[mask_with_booleans]
        #
        # probs = F.softmax(u_t)
        # attention_aware_state = torch.bmm(static_embeddings, probs.unsqueeze(2)).squeeze(2)
        # new_decoder_hidden = torch.cat((attention_aware_state, decoder_hidden_state), 1)
        # # to fernoume sto swsto dimension
        # new_decoder_hidden = self.tanh(self.hidden_out(new_decoder_hidden))
        # return probs, new_decoder_hidden.unsqueeze(0)