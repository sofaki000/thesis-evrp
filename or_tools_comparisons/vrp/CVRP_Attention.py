import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size), requires_grad=True))

        self._inf = nn.Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
    def init_inf(self, mask_size):
        # mask size: [bs, seq_len]
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)
    def forward(self, static_embedding, dynamic_embedding, decoder_output, mask):
        '''
        static_embeddings: ta embedings twn static features. Shape: [batch_size, hidden_size, seq_len]
        dynamic_embedding: ta embeddings twn dynamic features. Shape: [batch_size, hidden_size, seq_len]
        decoder_output: to decoder output. Shape: [batch_size, hidden_size]
        mask: True an apagoreuetai na pas, False an epitrepetai. Shape: [batch_size, seq_len]
        '''

        batch_size, hidden_size, _ = static_embedding.size()

        hidden =decoder_output.transpose(2,1).expand_as(static_embedding) #  # same as .expand(-1,-1,seq_len)
        hidden = torch.cat((static_embedding, dynamic_embedding, hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size) #v shape: [batch_size, 1, hidden_size]
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden))).squeeze(1)

        if len(attns[mask]) > 0:
            attns[mask] = self.inf[mask]  # osa exoun true, tous dinoume 0 probability

        logits = F.softmax(attns, dim=1)  # (batch, seq_len)
        return logits # [batch_size, seq_len]