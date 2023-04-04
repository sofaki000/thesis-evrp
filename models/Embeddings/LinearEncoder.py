
import torch
import torch.nn as nn

class LinearEncoder(nn.Module):
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