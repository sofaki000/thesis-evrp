import torch
import torch.nn as nn


class Env():
    def __init__(self, x, node_embeddings):
        super().__init__()
        """ depot_xy: (batch, 2)
            customer_xy: (batch, n_nodes-1, 2)
            --> self.xy: (batch, n_nodes, 2), Coordinates of depot + customer nodes
            demand: (batch, n_nodes-1)
            node_embeddings: (batch, n_nodes, embed_dim)
            is_next_depot: (batch, 1), e.g., [[True], [True], ...]
            Nodes that have been visited will be marked with True.
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.depot_xy, customer_xy, self.demand = x
        self.depot_xy, customer_xy, self.demand = self.depot_xy.to(self.device), customer_xy.to(
            self.device), self.demand.to(self.device)
        self.xy = torch.cat([self.depot_xy[:, None, :], customer_xy], 1).to(self.device)
        self.node_embeddings = node_embeddings
        self.batch, self.n_nodes, self.embed_dim = node_embeddings.size()

        self.is_next_depot = torch.ones([self.batch, 1], dtype=torch.bool).to(self.device)
        self.visited_customer = torch.zeros((self.batch, self.n_nodes - 1, 1), dtype=torch.bool).to(self.device)

    def get_mask_D(self, next_node, visited_mask, D):
        """ next_node: ([[0],[0],[not 0], ...], (batch, 1), dtype = torch.int32), [0] denotes going to depot
            visited_mask **includes depot**: (batch, n_nodes, 1)
            visited_mask[:,1:,:] **excludes depot**: (batch, n_nodes-1, 1)
            customer_idx **excludes depot**: (batch, 1), range[0, n_nodes-1] e.g. [[3],[0],[5],[11], ...], [0] denotes 0th customer, not depot
            self.demand **excludes depot**: (batch, n_nodes-1)
            selected_demand: (batch, 1)
            if next node is depot, do not select demand
            D: (batch, 1), D denotes "remaining vehicle capacity"
            self.capacity_over_customer **excludes depot**: (batch, n_nodes-1)
            visited_customer **excludes depot**: (batch, n_nodes-1, 1)
             is_next_depot: (batch, 1), e.g. [[True], [True], ...]
             return mask: (batch, n_nodes, 1)
        """
        self.is_next_depot = next_node == 0
        D = D.masked_fill(self.is_next_depot == True, 1.0)
        self.visited_customer = self.visited_customer | visited_mask[:, 1:, :]
        customer_idx = torch.argmax(visited_mask[:, 1:, :].type(torch.long), dim=1)
        selected_demand = torch.gather(input=self.demand, dim=1, index=customer_idx)
        D = D - selected_demand * (1.0 - self.is_next_depot.float())
        capacity_over_customer = self.demand > D
        mask_customer = capacity_over_customer[:, :, None] | self.visited_customer
        mask_depot = self.is_next_depot & (torch.sum((mask_customer == False).type(torch.long), dim=1) > 0)

        """ mask_depot = True
            ==> We cannot choose depot in the next step if 1) next destination is depot or 2) there is a node which has not been visited yet
        """
        return torch.cat([mask_depot[:, None, :], mask_customer], dim=1), D

    def _get_step(self, next_node, D):
        """ next_node **includes depot** : (batch, 1) int, range[0, n_nodes-1]
            --> one_hot: (batch, 1, n_nodes)
            node_embeddings: (batch, n_nodes, embed_dim)
            demand: (batch, n_nodes-1)
            --> if the customer node is visited, demand goes to 0
            prev_node_embedding: (batch, 1, embed_dim)
            context: (batch, 1, embed_dim+1)
        """
        one_hot = torch.eye(self.n_nodes)[next_node]
        visited_mask = one_hot.type(torch.bool).permute(0, 2, 1).to(self.device)

        mask, D = self.get_mask_D(next_node, visited_mask, D)
        self.demand = self.demand.masked_fill(self.visited_customer[:, :, 0] == True, 0.0)

        prev_node_embedding = torch.gather(input=self.node_embeddings, dim=1,
                                           index=next_node[:, :, None].repeat(1, 1, self.embed_dim))
        # prev_node_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = next_node[:,:,None].expand(self.batch,1,self.embed_dim))

        step_context = torch.cat([prev_node_embedding, D[:, :, None]], dim=-1)
        return mask, step_context, D

    def _create_t1(self):
        mask_t1 = self.create_mask_t1()
        step_context_t1, D_t1 = self.create_context_D_t1()
        return mask_t1, step_context_t1, D_t1

    def create_mask_t1(self):
        mask_customer = self.visited_customer.to(self.device)
        mask_depot = torch.ones([self.batch, 1, 1], dtype=torch.bool).to(self.device)
        return torch.cat([mask_depot, mask_customer], dim=1)

    def create_context_D_t1(self):
        D_t1 = torch.ones([self.batch, 1], dtype=torch.float).to(self.device)
        depot_idx = torch.zeros([self.batch, 1], dtype=torch.long).to(self.device)  # long == int64
        depot_embedding = torch.gather(input=self.node_embeddings, dim=1,
                                       index=depot_idx[:, :, None].repeat(1, 1, self.embed_dim))
        # depot_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = depot_idx[:,:,None].expand(self.batch,1,self.embed_dim))
        # https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        return torch.cat([depot_embedding, D_t1[:, :, None]], dim=-1), D_t1

    def get_log_likelihood(self, _log_p, pi):
        """ _log_p: (batch, decode_step, n_nodes)
            pi: (batch, decode_step), predicted tour
        """
        log_p = torch.gather(input=_log_p, dim=2, index=pi[:, :, None])
        return torch.sum(log_p.squeeze(-1), 1)

    def get_costs(self, pi):
        """ self.xy: (batch, n_nodes, 2), Coordinates of depot + customer nodes
            pi: (batch, decode_step), predicted tour
            d: (batch, decode_step, 2)
            Note: first element of pi is not depot, the first selected node in the path
        """
        d = torch.gather(input=self.xy, dim=1, index=pi[:, :, None].repeat(1, 1, 2))
        # d = torch.gather(input = self.xy, dim = 1, index = pi[:,:,None].expand(self.batch,pi.size(1),2))
        return (torch.sum((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2), dim=1)
                + (d[:, 0] - self.depot_xy).norm(p=2, dim=1)  # distance from depot to first selected node
                + (d[:, -1] - self.depot_xy).norm(p=2, dim=1)
                # distance from last selected node (!=0 for graph with longest path) to depot
                )


class Sampler(nn.Module):
    """ args; logits: (batch, n_nodes)
        return; next_node: (batch, 1)
        TopKSampler <=> greedy; sample one with biggest probability
        CategoricalSampler <=> sampling; randomly sample one from possible distribution based on probability
    """

    def __init__(self, n_samples=1, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples


class TopKSampler(Sampler):
    def forward(self, logits):
        return torch.topk(logits, self.n_samples, dim=1)[1]  # == torch.argmax(log_p, dim = 1).unsqueeze(-1)


class CategoricalSampler(Sampler):
    def forward(self, logits):
        return torch.multinomial(logits.exp(), self.n_samples)