import torch
import torch.nn as nn




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