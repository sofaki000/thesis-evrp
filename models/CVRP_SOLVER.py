import torch.nn as nn
from models.MHA_MODELS.MHA_model_cvrp import MHA_CVRP_solver
from or_tools_comparisons.vrp.cvrp_main import train_cvrp_model_pntr
from or_tools_comparisons.vrp.cvrp_model import CVRPSolver_PointerNetwork

from datasets.CVRP_dataset import   update_mask_cvrp, update_dynamic

# auth h classh dinei ena montelo pou lynei to CVRP. auth th
# stigmh exoume dyo montela: ena pou xrhsimopoiei multihead attention
# kai ena pou xrhsimopoiei plain pointer network.
from or_tools_comparisons.vrp.multihead_attention_model_train import train_model_with_multihead_attention


class CVRP_SOLVER_MODEL(nn.Module):
    def __init__(self, use_multihead_attention, use_pointer_network):
        super().__init__()
        self.use_multihead_attention = use_multihead_attention
        self.use_pointer_network = use_pointer_network


        if use_multihead_attention:
            print("Using multihead attention model...")
            self.model = MHA_CVRP_solver()

        elif use_pointer_network:
            print("Using pointer network model...")
            self.model = CVRPSolver_PointerNetwork(update_mask=update_mask_cvrp, update_dynamic=update_dynamic)

    def forward(self, static, dynamic,distance_matrix=None):
        if self.use_pointer_network:
            return self.model(static, dynamic)
        else:
            return self.model(static, dynamic,distance_matrix)



def get_trained_model_for_cvrp(use_multihead_attention, use_pointer_network,epochs, train_loader, validation_loader):
    # TODO: merge two train methods to one

    if use_pointer_network:
        model = CVRP_SOLVER_MODEL(use_multihead_attention=False, use_pointer_network=True)
        return train_cvrp_model_pntr(model, epochs, train_loader, validation_loader)

    elif use_multihead_attention:
        model = CVRP_SOLVER_MODEL(use_multihead_attention=True, use_pointer_network=False)
        return train_model_with_multihead_attention(model, epochs, train_loader, validation_loader)