import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.embedding = nn.Conv1d(in_channels=in_feats,
                                   out_channels=out_feats,
                                   kernel_size=1)
    def forward(self, input):
        '''
        :param input: [bs, in_feats, seq_len]
        :return: [bs, out_feats, seq_len]
        '''
        return self.embedding(input)

class Decoder(nn.Module):
    def __init__(self,in_feats,hidden_size):
        super().__init__()
        self.lstm= nn.LSTM(input_size=in_feats, hidden_size=hidden_size, batch_first=True)

    def forward(self, input, states):
        '''
        :param input: [batch_size, seq_len=1, features]
        :param states:
        :return:
        '''
        decoder_output, dec_states =  self.lstm(input, states)

        return decoder_output, dec_states