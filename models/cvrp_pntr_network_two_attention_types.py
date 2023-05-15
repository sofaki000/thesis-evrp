import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# source: https://github.com/higgsfield/np-hard-deep-reinforcement-learning/blob/master/Neural%20Combinatorial%20Optimization.ipynb
from datasets.CVRP_dataset import update_mask_cvrp_v2, update_dynamic
from or_tools_comparisons.CVRP.cvrp_model import Embedding
from ploting.plot_utilities import get_filename_time
from ploting.plot_utilities_attention import plot_attention_weights_heatmap_for_each_timestep, \
    get_next_experiment_number, get_attention_weights_dir

dynamic_features = 2
static_features = 2


#TODO: implement this attention to use dynamic state!
# 1st way: use ready attention
# 2nd way: see how query can be changed to include dynamic state
class Attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, C=10, name='Bahdanau'):
        super(Attention, self).__init__()

        self.use_tanh = use_tanh
        self.C = C
        self.name = name

        if name == 'Bahdanau':
            print("Using bahdanau attention...")
            self.W_query = nn.Linear(hidden_size, hidden_size)
            self.W_ref = nn.Conv1d(hidden_size, hidden_size, 1, 1)

            V = torch.FloatTensor(hidden_size)

            self.V = nn.Parameter(V)
            self.V.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))
        else:
            print("Using dot....")

        # question: How to add dynamic state as part of query OR ref?
        # As part of ref:
        # For attempt 1 we need this layer:
        # self.getRefFromEmbeddings =  nn.Linear(in_features=hidden_size*2, out_features=hidden_size)

        # For attempt 2 we need this layer:
        self.getQueryFromConcatenation = nn.Linear(in_features=hidden_size*2, out_features=hidden_size)
    def forward(self, query, last_decoder_output, dynamic_embeddings, static_embeddings):
        """
        Args:
            query: [batch_size x hidden_size]
            dynamic_embeddings: [ batch_size, seq_len, hidden_size]
            static_embeddings: [ batch_size, seq_len, hidden_size]
            ref:   [batch_size x seq_len x hidden_size]
        """

        # Attempt 1: ref is the concatenated dynamic + static embeddings
        # concatenated_embeddings = torch.cat((dynamic_embeddings, static_embeddings) , dim=2)
        # # ref must be [batch_size, seq_len, hidden_size]
        #
        # ref = self.getRefFromEmbeddings(concatenated_embeddings)

        # Attempt 2:
        # ref is the static embeddings
        ref = static_embeddings
        # query is the concatenation of previous output from decoder and current decoder state (hidden state)
        query = self.getQueryFromConcatenation(torch.cat((last_decoder_output , query), dim=1))

        batch_size = ref.size(0)
        seq_len = ref.size(1)

        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
            ref = self.W_ref(ref)  # [batch_size x hidden_size x seq_len]
            expanded_query = query.repeat(1, 1, seq_len)  # [batch_size x hidden_size x seq_len]
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size x 1 x hidden_size]
            logits = torch.bmm(V, F.tanh(expanded_query + ref)).squeeze(1)

        elif self.name == 'Dot':
            query = query.unsqueeze(2)
            # logits einai to probability to visit each index
            logits = torch.bmm(ref, query).squeeze(2)  # [batch_size x seq_len x 1]
            ref = ref.permute(0, 2, 1)
        else:
            raise NotImplementedError

        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        else:
            logits = logits

        return ref, logits # logits antistoixoun se probability_to_visit_each_index

class Attention_using_only_static_feats(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, C=10, name='Bahdanau'):
        super(Attention_using_only_static_feats, self).__init__()

        self.use_tanh = use_tanh
        self.C = C
        self.name = name

        if name == 'Bahdanau':
            print("Using bahdanau attention...")
            self.W_query = nn.Linear(hidden_size, hidden_size)
            self.W_ref = nn.Conv1d(hidden_size, hidden_size, 1, 1)

            V = torch.FloatTensor(hidden_size)

            self.V = nn.Parameter(V)
            self.V.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

    def forward(self, query, ref):
        """
        Args:
            query: [batch_size x hidden_size]
            ref:   ]batch_size x seq_len x hidden_size]
        """

        batch_size = ref.size(0)
        seq_len = ref.size(1)

        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
            ref = self.W_ref(ref)  # [batch_size x hidden_size x seq_len]
            expanded_query = query.repeat(1, 1, seq_len)  # [batch_size x hidden_size x seq_len]
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size x 1 x hidden_size]
            logits = torch.bmm(V, F.tanh(expanded_query + ref)).squeeze(1)

        elif self.name == 'Dot':
            query = query.unsqueeze(2)

            # mallon ta logits einai to probability to visit each index
            logits = torch.bmm(ref, query).squeeze(2)  # [batch_size x seq_len x 1]
            ref = ref.permute(0, 2, 1)
        else:
            raise NotImplementedError

        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        else:
            logits = logits

        return ref, logits #logits antistoixoun se probability_to_visit_each_index

# TODO: use this for embedding
class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(GraphEmbedding, self).__init__()
        self.embedding_size = embedding_size

        self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size))
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(2)
        embedding = self.embedding.repeat(batch_size, 1, 1)
        embedded = []
        inputs = inputs.unsqueeze(1)
        for i in range(seq_len):
            embedded.append(torch.bmm(inputs[:, :, :, i].float(), embedding))
        embedded = torch.cat(embedded, 1)
        return embedded

class PointerNet(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 attention_type,
                 experiment_name=None,
                 n_glimpses=5,
                 tanh_exploration=10,
                 use_tanh=False):
        super(PointerNet, self).__init__()

        self.experiment_name = experiment_name
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        #self.seq_len = seq_len

        self.embedding = Embedding(input_feats=static_features, out_feats=hidden_size)
        self.dynamic_embedding = Embedding(input_feats=dynamic_features, out_feats=hidden_size)

        #TODO: understand what this layer does
        #nn.Embedding(embedding_size, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)

        print(f'USING ATTENTION {attention_type}')
        self.pointer = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration , name = attention_type)
        self.glimpse = Attention(hidden_size, use_tanh=False ,name = attention_type )

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

        # self.criterion = nn.CrossEntropyLoss()
        self.max_steps = 5 #1000

        self.embedding_size = embedding_size

        experiment_folder_name = f"exp{get_next_experiment_number(get_attention_weights_dir())}"

        self.experiment_folder_name = experiment_folder_name
    def apply_mask_to_logits(self, mask, dynamic,logits, chosen_idx):
        # oi maskes einai 0 kai 1. 1 mporeis na pas, 0 den mporeis na pas

        mask = update_mask_cvrp_v2(dynamic, chosen_idx ).byte()

        batch_size = dynamic.size(0)
        #
        if chosen_idx is not None:
            mask[[i for i in range(batch_size)], chosen_idx.data] = 0 # opou phgame (0) den mporoume na xanapame
            logits[mask.eq(0)] = -np.inf

        if (mask.sum(dim=1) == 0).any(): # means some routes are done some are not
            mask[(mask.sum(dim=1) == 0), 0] = 1 # se autes epitrepoume na pane sto depot
        return logits, mask

          # creates an exp{number} folder to store next experiments results


    def forward(self, static, dynamic, current_epoch=None):
        # TODO: add experiment how dynamic information changes model
        #TODO: OMG ADD DYNAMIC DYMENSION!!
        """ Args:  inputs: [batch_size x sourceL]  """
        batch_size = static.size(0)
        seq_len = static.size(2)

        static_embedding = self.embedding(static)

        dynamic_embedding = self.dynamic_embedding(dynamic).transpose(2,1)

        encoder_outputs, (hidden, context) = self.encoder(static_embedding.transpose(2,1))

        mask = torch.ones(batch_size, seq_len).byte()

        chosen_indexes = None

        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        attention_weights_at_each_timestep  = []

        tours = []
        tour_logp = []

        for i in range(self.max_steps):
            if not mask.byte().any(): # if can't visit any more indexes, finish it
                break

            last_decoder_output, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            last_decoder_output = last_decoder_output.squeeze(1)
            query = hidden.squeeze(0) # query is the last output from decoder + current decoder
            # state we will later add!

            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(query, last_decoder_output, dynamic_embedding, encoder_outputs)
                ## TODO : understand glimpses!!!
                #TODO: do we need to update logits here?
                # logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                # logits, mask = self.apply_mask_to_logits(mask, dynamic,logits, chosen_indexes)
                query = torch.bmm(ref, F.softmax(logits).unsqueeze(2)).squeeze(2)

            _, logits = self.pointer(query,last_decoder_output, dynamic_embedding, encoder_outputs)


            # probably we should use these logits to chose indexes

            logits, mask = self.apply_mask_to_logits(mask,dynamic, logits, chosen_indexes)

            if not mask.byte().any():
                # if should_terminate_cvrp(dynamic):
                break

            # decoder_input = embedded[:, i, :] #target_embedded[:, i, :]

            # assert decoder_input.size(0) == batch_size and decoder_input.size(1) == 1
            # assert decoder_input.size(2) == self.embedding_size
            # this is for supervised learning. we want rl
            # loss += self.criterion(logits, target[:, i])
            # based on the logits we have, we calculate the posibility to visit each index
            probability_to_visit_each_vertex = F.softmax(logits + mask.log(), dim=1)  # (batch, seq_len)


            ####
            m = torch.distributions.Categorical(probability_to_visit_each_vertex)
            ptr = m.sample()
            chosen_indexes = ptr.data.detach()
            logp = m.log_prob(ptr)

            dynamic = update_dynamic(dynamic, chosen_indexes)

            #### for plotting attention weights
            attention_weights_at_current_time_step = probability_to_visit_each_vertex[0].detach().numpy() # logits[0].detach().numpy()
            attention_weights_at_each_timestep.append(attention_weights_at_current_time_step)

            # we update dynamic input based on chosen indexes
            decoder_input = static_embedding[:, :, chosen_indexes.long()] [:,:,0]  #embedded[:, chosen_indexes.long(), :][:, 0, :]

            # we store the result
            tour_logp.append(logp.unsqueeze(1))
            tours.append(chosen_indexes.unsqueeze(1)) # chosen_indexes: [batch_size]

        #they were returning: return loss / seq_len

        tours = torch.cat(tours, 1)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)


        if current_epoch is not None:
            # plot attention weights at each time step for the current epoch.
            # the goal is to see how attention changes with the training for the same data.
            plot_attention_weights_heatmap_for_each_timestep(attention_weights_at_each_timestep,
                                                             self.experiment_name ,
                                                             self.experiment_folder_name,
                                                             current_epoch)

        return tours, tour_logp




