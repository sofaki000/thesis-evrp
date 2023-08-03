import math

import torch
import torch.nn as nn

from data import generate_data
from decoder_utils import TopKSampler, CategoricalSampler, Env

class DotProductAttention(nn.Module):
	def __init__(self, clip = None, return_logits = False, head_depth = 16, inf = 1e+10, **kwargs):
		super().__init__(**kwargs)
		self.clip = clip
		self.return_logits = return_logits
		self.inf = inf
		self.scale = math.sqrt(head_depth)
		# self.tanh = nn.Tanh()

	def forward(self, x, mask = None):
		""" Q: (batch, n_heads, q_seq(=n_nodes or =1), head_depth)
			K: (batch, n_heads, k_seq(=n_nodes), head_depth)
			logits: (batch, n_heads, q_seq(this could be 1), k_seq)
			mask: (batch, n_nodes, 1), e.g. tf.Tensor([[ True], [ True], [False]])
			mask[:,None,None,:,0]: (batch, 1, 1, n_nodes) ==> broadcast depending on logits shape
			[True] -> [1 * -np.inf], [False] -> [logits]
			K.transpose(-1,-2).size() == K.permute(0,1,-1,-2).size()
		"""
		Q, K, V = x
		logits = torch.matmul(Q, K.transpose(-1,-2)) / self.scale
		if self.clip is not None:
			logits = self.clip * torch.tanh(logits)

		if self.return_logits:
			if mask is not None:
				return logits.masked_fill(mask.permute(0,2,1) == True, -self.inf)
			return logits

		if mask is not None:
			logits = logits.masked_fill(mask[:,None,None,:,0].repeat(1,logits.size(1),1,1) == True, -self.inf)

		probs = torch.softmax(logits, dim = -1)
		return torch.matmul(probs, V)
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=8, embed_dim=128, clip=None, return_logits=None, need_W=None):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_depth = self.embed_dim // self.n_heads
        if self.embed_dim % self.n_heads != 0:
            raise ValueError("embed_dim = n_heads * head_depth")

        self.need_W = need_W
        self.attention = DotProductAttention(clip=clip, return_logits=return_logits, head_depth=self.head_depth)
        if self.need_W:
            self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
            self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
            self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
            self.Wout = nn.Linear(embed_dim, embed_dim, bias=False)
        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            if name == 'Wout.weight':
                stdv = 1. / math.sqrt(param.size(-1))
            elif name in ['Wk.weight', 'Wv.weight', 'Wq.weight']:
                stdv = 1. / math.sqrt(self.head_depth)
            else:
                raise ValueError
            param.data.uniform_(-stdv, stdv)

    def split_heads(self, T):
        """ https://qiita.com/halhorn/items/c91497522be27bde17ce
            T: (batch, n_nodes, self.embed_dim)
            T reshaped: (batch, n_nodes, self.n_heads, self.head_depth)
            return: (batch, self.n_heads, n_nodes, self.head_depth)

            https://raishi12.hatenablog.com/entry/2020/04/20/221905
        """
        shape = T.size()[:-1] + (self.n_heads, self.head_depth)
        T = T.view(*shape)
        return T.permute(0, 2, 1, 3)

    def combine_heads(self, T):
        """ T: (batch, self.n_heads, n_nodes, self.head_depth)
            T transposed: (batch, n_nodes, self.n_heads, self.head_depth)
            return: (batch, n_nodes, self.embed_dim)
        """
        T = T.permute(0, 2, 1, 3).contiguous()
        shape = T.size()[:-2] + (self.embed_dim,)
        return T.view(*shape)

    def forward(self, x, mask=None):
        """	q, k, v = x
            encoder arg x: [x, x, x]
            shape of q: (batch, n_nodes, embed_dim)
            output[0] - output[h_heads-1]: (batch, n_nodes, head_depth)
            --> concat output: (batch, n_nodes, head_depth * h_heads)
            return output: (batch, n_nodes, embed_dim)
        """
        Q, K, V = x
        if self.need_W:
            Q, K, V = self.Wq(Q), self.Wk(K), self.Wv(V)
        Q, K, V = list(map(self.split_heads, [Q, K, V]))
        output = self.attention([Q, K, V], mask=mask)
        output = self.combine_heads(output)
        if self.need_W:
            return self.Wout(output)
        return output
class DecoderCell(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8, clip=10., **kwargs):
        super().__init__(**kwargs)

        self.Wk1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_fixed = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wout = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_step = nn.Linear(embed_dim + 1, embed_dim, bias=False)

        self.MHA = MultiHeadAttention(n_heads=n_heads, embed_dim=embed_dim, need_W=False)
        self.SHA = DotProductAttention(clip=clip, return_logits=True, head_depth=embed_dim)
        # SHA ==> Single Head Attention, because this layer n_heads = 1 which means no need to spilt heads
        self.env = Env

    def compute_static(self, node_embeddings, graph_embedding):
        self.Q_fixed = self.Wq_fixed(graph_embedding[:, None, :])
        self.K1 = self.Wk1(node_embeddings)
        self.V = self.Wv(node_embeddings)
        self.K2 = self.Wk2(node_embeddings)

    def compute_dynamic(self, mask, step_context):
        Q_step = self.Wq_step(step_context)
        Q1 = self.Q_fixed + Q_step
        Q2 = self.MHA([Q1, self.K1, self.V], mask=mask)
        Q2 = self.Wout(Q2)
        logits = self.SHA([Q2, self.K2, None], mask=mask)
        return logits.squeeze(dim=1)

    def forward(self, x, encoder_output, return_pi=False, decode_type='sampling'):
        node_embeddings, graph_embedding = encoder_output
        self.compute_static(node_embeddings, graph_embedding)
        env = Env(x, node_embeddings)
        mask, step_context, D = env._create_t1()

        selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
        log_ps, tours = [], []
        for i in range(env.n_nodes * 2):
            logits = self.compute_dynamic(mask, step_context)
            log_p = torch.log_softmax(logits, dim=-1)
            next_node = selecter(log_p)
            mask, step_context, D = env._get_step(next_node, D)
            tours.append(next_node.squeeze(1))
            log_ps.append(log_p)
            if env.visited_customer.all():
                break

        pi = torch.stack(tours, 1)
        cost = env.get_costs(pi)
        ll = env.get_log_likelihood(torch.stack(log_ps, 1), pi)

        if return_pi:
            return cost, ll, pi
        return cost, ll


if __name__ == '__main__':
    batch, n_nodes, embed_dim = 5, 21, 128
    data = generate_data(n_samples=batch, n_customer=n_nodes - 1)
    decoder = DecoderCell(embed_dim, n_heads=8, clip=10.)
    node_embeddings = torch.rand((batch, n_nodes, embed_dim), dtype=torch.float)
    graph_embedding = torch.rand((batch, embed_dim), dtype=torch.float)
    encoder_output = (node_embeddings, graph_embedding)
    # a = graph_embedding[:,None,:].expand(batch, 7, embed_dim)
    # a = graph_embedding[:,None,:].repeat(1, 7, 1)
    # print(a.size())

    decoder.train()
    cost, ll, pi = decoder(data, encoder_output, return_pi=True, decode_type='sampling')
    print('\ncost: ', cost.size(), cost)
    print('\nll: ', ll.size(), ll)
    print('\npi: ', pi.size(), pi)

# cnt = 0
# for i, k in decoder.state_dict().items():
# 	print(i, k.size(), torch.numel(k))
# 	cnt += torch.numel(k)
# print(cnt)

# ll.mean().backward()
# print(decoder.Wk1.weight.grad)
# https://discuss.pytorch.org/t/model-param-grad-is-none-how-to-debug/52634