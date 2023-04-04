import torch
import torch.nn as nn

from models.MHA_MODELS.MHA_model import MHA_EVRP_solver


class EVRP_SOLVER(nn.Module):
    def __init__(self):
        super().__init__()

        self.solver = MHA_EVRP_solver()

    def forward(self, static, dynamic,distances):

        return self.solver(static, dynamic,distances)


