import torch.nn as nn
import torch
import math

import torch.nn.functional as F
class Attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, C=10):
        super().__init__()

        self.use_tanh = use_tanh
        self.C = C
        self.W_query = nn.Linear(hidden_size, hidden_size)
        self.W_ref = nn.Conv1d(hidden_size, hidden_size,1,1)

        V = torch.FloatTensor(hidden_size)
        self.V = nn.Parameter(V)
        self.V.data.uniform_(-(1. / math.sqrt(hidden_size)) , 1. / math.sqrt(hidden_size))

    def forward(self, decoder_hidden, encoder_outputs):
        '''
        decoder_hidden: QUERY  [batch_size x hidden_size]
        encoder_outputs: REF [batch_size x seq_len x hidden_size]
        returns: logits: [ batch_size, seq_len]
        '''

        query = decoder_hidden.squeeze()
        ref = encoder_outputs

        batch_size = ref.size(0)
        seq_len = ref.size(1)

        ref = ref.permute(0,2,1)

        if batch_size==1:
            query = self.W_query(query).unsqueeze(1).unsqueeze(2).transpose(1,0)
        else:
            query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]

        ref = self.W_ref(ref)  # [batch_size x hidden_size x seq_len]

        expanded_query = query.repeat(1,1,seq_len) # [batch_size, hidden_size, seq_len]
        V = self.V.unsqueeze(0).repeat(batch_size,1,1) # [ batch_size, 1, hidden_size]

        logits = torch.bmm(V,F.tanh(expanded_query+ ref)).squeeze(1)

        if self.use_tanh:
            logits = self.C * F.tanh(logits)


        return logits