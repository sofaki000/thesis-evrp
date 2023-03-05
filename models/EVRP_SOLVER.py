import torch.nn as nn
from models.MHA_MODELS.MHA_model_cvrp import MHA_CVRP_solver


class EVRP_SOLVER_MODEL(nn.Module):
    def __init__(self, use_multihead_attention):
        super().__init__()
        self.use_multihead_attention = use_multihead_attention


        if use_multihead_attention:
            self.model = MHA_CVRP_solver()

    def forward(self, static, dynamic,distance_matrix):
        return self.model(static, dynamic,distance_matrix)