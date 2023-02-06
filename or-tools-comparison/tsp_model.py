import numpy as np

from AttentionModel import Attention
import torch.nn.functional as F
import math
import torch.nn as nn
import torch

static_features = 2
hidden_size = 128

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(input_size=static_features, hidden_size=hidden_size, batch_first=True)

    def forward(self, input):
        '''
        input: [batch_size, num_nodes (sequence_length), static_features]
        '''

        return self.encoder(input)

class Decoder(nn.Module):
    def __init__(self,sequence_length):
        super().__init__()

        self.pointer = Attention(hidden_size)
        self.decoder = nn.LSTM(input_size= hidden_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=sequence_length)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(hidden_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

        self.embedding = nn.Linear(in_features=1, out_features=hidden_size)

        self.criterion = nn.CrossEntropyLoss()

    def apply_mask_to_logits(self, logits, mask, idexes):
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idexes is not None:
            clone_mask[[i for i in range(batch_size)], idexes.data] = 1

            logits[clone_mask] = -np.inf
        else:
            logits[:, :] = -np.inf
            logits[:, 0] = 1
        return logits, clone_mask
    def forward(self, encoder_outputs, inputs, targets ,encoder_hidden_states, encoder_context):

        batch_size = inputs.size(0)
        seq_len = inputs.size(1)

        time_steps = seq_len

        hidden, context = (encoder_hidden_states, encoder_context)

        # xekiname apo ena uniform initialized parameter me ones
        decoder_input = self.embedding(torch.ones(batch_size,1,1)) #self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1).unsqueeze(2).transpose(2,1)

        # we calculate loss per batch here
        loss = 0

        # outputs hold the predictions we made for each timestep in the batch samples. has size
        # batch_size, time_steps (aka sequence length)
        outputs = torch.zeros(time_steps, batch_size)

        tours = []

        mask = torch.zeros(batch_size, seq_len).byte()
        idxs = None

        # gia kathe time step, briskoume tis pithanothtes na episkeutei ta epomena
        for time_step in range(time_steps):

            assert decoder_input.size(0) == batch_size and decoder_input.size(1) == 1 and decoder_input.size(2) == hidden_size

            decoder_output, (hidden, context) = self.decoder(decoder_input,(hidden, context))

            logits = self.pointer(hidden, encoder_outputs)

            if batch_size==1:
                loss += self.criterion(logits , targets[time_step].unsqueeze(0))
            else:
                loss += self.criterion(logits,targets[:,time_step])

            masked_logits, mask = self.apply_mask_to_logits(logits, mask, idxs)

            probabilities_to_visit_each_node = F.softmax(masked_logits, dim=1)
            idxs = probabilities_to_visit_each_node.argmax(1)
            # outputs[time_step] = idxs

            tours.append(idxs.unsqueeze(1))  # [batch_size] -> [bs,1]
            if batch_size==1:
                decoder_input =self.embedding(targets[time_step].unsqueeze(0).unsqueeze(1).float()).unsqueeze(1)
            else:
                decoder_input = self.embedding(targets[:,time_step].unsqueeze(1).float()).unsqueeze(1)

        tours = torch.cat(tours, 1)
        loss /= batch_size
        return loss,tours

class TSP_model(nn.Module):
    def __init__(self,sequence_length):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(sequence_length)
    def forward(self, inputs, targets):
        encoded, (hidden_states, context) = self.encoder(inputs)

        loss, tours = self.decoder(encoded, inputs, targets,hidden_states, context)
        return loss , tours