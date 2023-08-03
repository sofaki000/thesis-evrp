import torch.nn as nn
from models.MHA_MODELS.MHA_model_cvrp import MHA_CVRP_solver
from models.cvrp_pntr_network_two_attention_types import PointerNet
from or_tools_comparisons.CVRP.cvrp_main import train_cvrp_model_pntr
from or_tools_comparisons.CVRP.cvrp_model import CVRPSolver_PointerNetwork

from datasets.CVRP_dataset import   update_mask_cvrp, update_dynamic

# auth h classh dinei ena montelo pou lynei to CVRP. auth th
# stigmh exoume dyo montela: ena pou xrhsimopoiei multihead attention
# kai ena pou xrhsimopoiei plain pointer network.
from or_tools_comparisons.CVRP.multihead_attention_model_train import train_model_with_multihead_attention


class CVRP_SOLVER_MODEL(nn.Module):
    def __init__(self, use_multihead_attention,
                 use_pointer_network,
                 use_pntr_with_attention_variations,
                 attention_type=None,
                 experiment_name=None,
                 attention_weights_photos_store_folder=None):
        super().__init__()
        self.use_multihead_attention = use_multihead_attention
        self.use_pointer_network = use_pointer_network
        self.use_pntr_with_attention_variations =use_pntr_with_attention_variations
        self.experiment_name = None
        if experiment_name is not None:
            self.experiment_name = experiment_name


        ## TODO: add official docs pointer network

        if use_pntr_with_attention_variations:
            print("Using new pointer network model...")
            self.model = PointerNet(embedding_size=128,
                                    hidden_size=128,
                                    experiment_name=self.experiment_name,
                                    attention_type=attention_type,
                                    attention_weights_photos_store_folder=attention_weights_photos_store_folder)
        elif use_multihead_attention:
            print("Using multihead attention model...")
            self.model = MHA_CVRP_solver()

        elif use_pointer_network:
            print("Using pointer network model...")
            self.model = CVRPSolver_PointerNetwork(update_mask=update_mask_cvrp,
                                                   update_dynamic=update_dynamic)

    def forward(self, static, dynamic,distance_matrix=None):

        if self.use_pntr_with_attention_variations:
            #TODO: refactor. here distance_matrix is number_of_epochs
            return self.model(static, dynamic,distance_matrix)
        elif self.use_pointer_network:
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


