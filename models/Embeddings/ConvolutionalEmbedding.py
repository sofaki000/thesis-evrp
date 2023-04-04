import torch.nn as nn

class ConvolutionalEncoder(nn.Module):
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