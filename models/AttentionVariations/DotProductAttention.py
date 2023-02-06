import torch
import torch.nn as nn
import math



class DotProductAttention(nn.Module):
    def __init__(self, clip=None, return_logits=False, head_depth=16, inf=1e+10, **kwargs):
        super().__init__(**kwargs)
        self.clip = clip
        self.return_logits = return_logits
        self.inf = inf
        self.scale = math.sqrt(head_depth)

    # self.tanh = nn.Tanh()

    def forward(self, x, mask=None):
        """ Q: (batch, n_heads, q_seq(=n_nodes or =1), head_depth)
            K: (batch, n_heads, k_seq(=n_nodes), head_depth)
            logits: (batch, n_heads, q_seq(this could be 1), k_seq)
            mask: (batch, n_nodes, 1), e.g. tf.Tensor([[ True], [ True], [False]])
            mask[:,None,None,:,0]: (batch, 1, 1, n_nodes) ==> broadcast depending on logits shape
            [True] -> [1 * -np.inf], [False] -> [logits]
            K.transpose(-1,-2).size() == K.permute(0,1,-1,-2).size()
        """
        Q, K, V = x
        logits = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        if self.clip is not None:
            logits = self.clip * torch.tanh(logits)

        if self.return_logits:
            if mask is not None:
                return logits.masked_fill(mask.permute(0, 2, 1) == True, -self.inf)
            return logits

        if mask is not None:
            logits = logits.masked_fill(mask[:, None, None, :, 0].repeat(1, logits.size(1), 1, 1) == True, -self.inf)

        probs = torch.softmax(logits, dim=-1)
        return torch.matmul(probs, V)