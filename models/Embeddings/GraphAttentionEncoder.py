
import torch
import torch.nn as nn
# from torchsummary import summary

import math

from models.AttentionVariations.Multihead import MultiHeadAttention

use_separate_embedding_for_each_dynamic_feature = True

CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}
def generate_data(n_samples=10, n_customer=20, seed=None):
    """ https://pytorch.org/docs/master/torch.html?highlight=rand#torch.randn
        x[0] -- depot_xy: (batch, 2)
        x[1] -- customer_xy: (batch, n_nodes-1, 2)
        x[2] -- demand: (batch, n_nodes-1)
    """
    if seed is not None:
        torch.manual_seed(seed)

    return (torch.rand((n_samples, 2) ),
            torch.rand((n_samples, n_customer, 2) ),
            torch.randint(size=(n_samples, n_customer), low=1, high=10 ) / CAPACITIES[n_customer])

class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super().__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d}.get(normalization, None)
        self.normalizer = normalizer_class(embed_dim, affine=True)

    # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
    # 	self.init_parameters()

    # def init_parameters(self):
    # 	for name, param in self.named_parameters():
    # 		stdv = 1. / math.sqrt(param.size(-1))
    # 		param.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            # (batch, num_features)
            # https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())

        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return x


class ResidualBlock_BN(nn.Module):
    def __init__(self, MHA, BN, **kwargs):
        super().__init__(**kwargs)
        self.MHA = MHA
        self.BN = BN

    def forward(self, x, mask=None):
        if mask is None:
            return self.BN(x + self.MHA(x))
        return self.BN(x + self.MHA(x, mask))


class SelfAttention(nn.Module):
    def __init__(self, MHA, **kwargs):
        super().__init__(**kwargs)
        self.MHA = MHA

    def forward(self, x, mask=None):
        return self.MHA([x, x, x], mask=mask)


class EncoderLayer(nn.Module):
    # nn.Sequential):
    def __init__(self, n_heads=8, FF_hidden=512, embed_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.FF_hidden = FF_hidden
        self.BN1 = Normalization(embed_dim, normalization='batch')
        self.BN2 = Normalization(embed_dim, normalization='batch')

        self.MHA_sublayer = ResidualBlock_BN(SelfAttention(MultiHeadAttention(n_heads=self.n_heads, embed_dim=embed_dim, need_W=True)), self.BN1)

        self.FF_sublayer = ResidualBlock_BN(
            nn.Sequential(
                nn.Linear(embed_dim, FF_hidden, bias=True),
                nn.ReLU(),
                nn.Linear(FF_hidden, embed_dim, bias=True)
            ),
            self.BN2 )

    def forward(self, x, mask=None):
        """	arg x: (batch, n_nodes, embed_dim)
            return: (batch, n_nodes, embed_dim)
        """
        return self.FF_sublayer(self.MHA_sublayer(x, mask=mask))


class GraphAttentionEncoder(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8, n_layers=3, FF_hidden=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.init_W_depot = torch.nn.Linear(2, embed_dim, bias=True)
        self.init_W_afs = torch.nn.Linear(2, embed_dim, bias=True)
        self.init_W = torch.nn.Linear(2, embed_dim, bias=True)
        self.encoder_layers = nn.ModuleList([EncoderLayer(n_heads, FF_hidden, embed_dim) for _ in range(n_layers)])

        self.init_time_duration = nn.Linear(1, embed_dim, bias=True)
        self.init_capacity = nn.Linear(1, embed_dim, bias=True)
        self.init_demands = nn.Linear(1, embed_dim, bias=True)

        # twra auto den xerw kata poso einai swsto na kanoume concatenate kai ta dynamic
        # embeddings kai anagkastika na ta kanoume concatenate sthn dim=2.
        # ta static embeddings ta kaname sthn dim=1 (sto num seq level).
        self.dynamic_embeddings_to_correct_size = nn.Linear(embed_dim*3, embed_dim, bias=True)

        self.dynamic_and_static_to_correct_size = nn.Linear(embed_dim * 2, embed_dim, bias=True)
    def forward(self, static, dynamic, mask=None):
        '''

        '''


        ### DYNAMIC EMBEDDINGS
        if use_separate_embedding_for_each_dynamic_feature: # we can embed each dynamic feature with different embedding (linear) layer
            time_duration = dynamic[:,0,:]
            time_duration_embed = self.init_time_duration(time_duration.unsqueeze(2))
            capacity = dynamic[:,1,:]
            capacity_embed = self.init_capacity(capacity.unsqueeze(2))
            demands = dynamic[:,2,:]
            demands_embed = self.init_demands(demands.unsqueeze(2))

            dynamic_embeddings = torch.cat((capacity_embed,demands_embed, time_duration_embed), dim=2)

            # we return dynamic embeddings to correct size
            dynamic_embeddings = self.dynamic_embeddings_to_correct_size(dynamic_embeddings)
        else:
            dynamic_embeddings = self.dynamic_embedding(dynamic) # apo [bs,feats,num_nodes] -> [bs, hidden_size, num_nodes]

        ########### STATIC EMBEDDINGS:
        depot_feats = static[:, :, 0] # [batch_size, 2]
        afs = static[:, :, 1:4] # [ bs, 2, num_afs]
        customers = static[:, :, 4:] # [bs, 2, num_customers]

        depot_embedding = self.init_W_depot(depot_feats)
        afs_embedding =  self.init_W_afs(afs.transpose(2,1)).transpose(2,1)
        customers_embeddings = self.init_W(customers.transpose(2,1)).transpose(1,2)


        assert depot_embedding.size(1)== self.embed_dim and afs_embedding.size(1) == self.embed_dim and customers_embeddings.size(1) ==self.embed_dim

        # static_embeddings: [bs, hidden_size, all_num_nodes]
        static_embeddings = torch.cat((afs_embedding, customers_embeddings, depot_embedding.unsqueeze(2)), dim=2)

        ###### WE CONCATENATE STATIC AND DYNAMIC EMBEDDINGS
        dynamic_and_static_embeddings = torch.cat((dynamic_embeddings.transpose(2,1), static_embeddings), dim=1)
        dynamic_and_static_embeddings = self.dynamic_and_static_to_correct_size(dynamic_and_static_embeddings.transpose(2,1))

        #static = torch.cat([self.init_W_depot(static[0])[:, None, :], self.init_W(torch.cat([static[1], static[2][:, :, None]], dim=-1))], dim=1)

        for layer in self.encoder_layers:
            static = layer(dynamic_and_static_embeddings, mask)

        return (static, torch.mean(static, dim=1))


if __name__ == '__main__':
    batch = 5
    n_nodes = 21
    encoder = GraphAttentionEncoder(n_layers=1)
    data = generate_data(n_samples=batch, n_customer=n_nodes - 1)
    # mask = torch.zeros((batch, n_nodes, 1), dtype = bool)
    output = encoder(data, mask=None)
    print('output[0].shape:', output[0].size())
    print('output[1].shape', output[1].size())

    # summary(encoder, [(2), (20,2), (20)])
    cnt = 0
    for i, k in encoder.state_dict().items():
        print(i, k.size(), torch.numel(k))
        cnt += torch.numel(k)
    print(cnt)

# output[0].mean().backward()
# print(encoder.init_W_depot.weight.grad)