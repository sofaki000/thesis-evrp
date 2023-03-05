import torch
import torch.nn as nn

from models.Embeddings.GraphEmbeddings.GraphAttentionEncoder import EncoderLayer


CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}

### TODO: after some point weights become nans!

class GraphAttentionEncoderForCVRP(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8, n_layers=1, FF_hidden=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_seperate_embedding_for_static_feats = False #True
        self.use_separate_embedding_for_each_dynamic_feature = False #  True #

        # layers to initialize static features
        if self.use_seperate_embedding_for_static_feats:
            self.init_W_depot = torch.nn.Linear(2, embed_dim, bias=True)
            self.init_W = torch.nn.Linear(2, embed_dim, bias=True)
        else:
            self.init_static = torch.nn.Linear(2, embed_dim)

        if self.use_separate_embedding_for_each_dynamic_feature:
            # self.init_time_duration = nn.Linear(1, embed_dim, bias=True)
            self.dynami_embedding_capacity = nn.Linear(1, embed_dim, bias=True)
            self.dynamic_embedding_demands = nn.Linear(1, embed_dim, bias=True)
        else:
            self.dynamic_embedding = nn.Linear(2, embed_dim, bias=True)



        #self.encoder_layers = nn.ModuleList([EncoderLayer(n_heads, FF_hidden, embed_dim) for _ in range(n_layers)])
        self.encoder_layer = EncoderLayer(n_heads, FF_hidden, embed_dim)

        self.dynamic_embeddings_to_correct_size = nn.Linear(embed_dim * 2, embed_dim, bias=True)

        self.dynamic_and_static_to_correct_size = nn.Linear(embed_dim * 2, embed_dim, bias=True)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic, mask=None):
        '''

        '''
        batch_size = static.size(0)
        seq_len = static.size(2)

        ### DYNAMIC EMBEDDINGS
        if self.use_separate_embedding_for_each_dynamic_feature:  # we can embed each dynamic feature with different embedding (linear) layer: problem: an den xeroume ta dynamics apo prin?

            capacity = dynamic[:, 0, :]
            capacity_embed = self.dynami_embedding_capacity(capacity.unsqueeze(2))
            demands = dynamic[:, 1, :]
            demands_embed = self.dynamic_embedding_demands(demands.unsqueeze(2))

            dynamic_embeddings = torch.cat((capacity_embed, demands_embed), dim=2)

            # we return dynamic embeddings to correct size
            dynamic_embeddings = self.dynamic_embeddings_to_correct_size(dynamic_embeddings).transpose(1, 2)
        else:
            assert dynamic.size(0) == batch_size
            assert dynamic.size(2) == seq_len
            # apo [bs,feats,num_nodes] -> [bs, hidden_size, num_nodes]
            #print(self.dynamic_embedding.weight)
            #print(self.dynamic_embedding.bias)
            dynamic_embeddings = self.dynamic_embedding(dynamic.transpose(1, 2)).transpose(1, 2)

            print(self.dynamic_embedding.weight)
        ########### STATIC EMBEDDINGS:

        if self.use_seperate_embedding_for_static_feats:
            depot_feats = static[:, :, 0]  # [batch_size, 2]
            customers = static[:, :, 1:]  # [bs, 2, num_customers]

            depot_embedding = self.init_W_depot(depot_feats)
            customers_embeddings = self.init_W(customers.transpose(2, 1)).transpose(1, 2)
            assert depot_embedding.size(1) == self.embed_dim and customers_embeddings.size(1) == self.embed_dim

            # static_embeddings: [bs, hidden_size, all_num_nodes]
            static_embeddings = torch.cat((customers_embeddings, depot_embedding.unsqueeze(2)), dim=2)
        else:
            static_embeddings = self.init_static(static.transpose(1, 2)).transpose(2, 1)

        assert static_embeddings.size(0) == batch_size and static_embeddings.size(2) == seq_len

        ###### WE CONCATENATE STATIC AND DYNAMIC EMBEDDINGS

        concatenate_dynamic_with_static_embeddings = False

        if concatenate_dynamic_with_static_embeddings:
            # FOR some reason doing this leads to bugs. Whereas in EVRP it doenst lead to bug
            dynamic_and_static_embeddings = torch.cat((dynamic_embeddings, static_embeddings), dim=1)
            final_embeddings = self.dynamic_and_static_to_correct_size(dynamic_and_static_embeddings.transpose(2, 1))
        else:
            final_embeddings = dynamic_embeddings.transpose(2,1)

        # #TODO: is mask ok? the same way in other model?
        # for layer in self.encoder_layers:
        #     static = layer(final_embeddings, mask)
        static = self.encoder_layer(final_embeddings, mask)

        return (static, torch.mean(static, dim=1))