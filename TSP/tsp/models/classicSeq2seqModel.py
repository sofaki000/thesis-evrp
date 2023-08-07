import numpy as np
from scipy.spatial import distance_matrix
import torch.nn.functional as F
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from or_tools_comparisons.common_utilities import get_tour_length_from_distance_matrix
from ploting.metrics.plot_average_tour_length import plot_average_tour_length
from ploting.metrics.plot_tour import format_tours, print_tensor
from ploting.plot_utilities import plot_train_and_validation_loss

static_features = 2
hidden_size = 128


## TODO: bug that model returns the same routes for every problem!!

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.RNN(input_size=static_features,
                              hidden_size=hidden_size,
                              batch_first=True)

    def forward(self, input):
        '''
        input: [batch_size, num_nodes (sequence_length), static_features]
        '''
        return self.encoder(input)



class Decoder(nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        self.decoder = nn.RNN(input_size=hidden_size,
                              hidden_size=hidden_size,
                              batch_first=True)


        self.decoderOutputToActualProbabilities = nn.Linear(hidden_size,sequence_length)

        self.targetToDecoderInput = nn.Linear(1, hidden_size)
        self.criterion = nn.CrossEntropyLoss()
    def apply_mask_to_logits(self, logits, mask, indexes):
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if indexes is not None:
            clone_mask[[i for i in range(batch_size)], indexes.data.squeeze(1).long()] = 1

            logits[clone_mask.unsqueeze(1)] = -np.inf
        else:
            logits[:, :] = -np.inf
            # we want to start from depot, ie the first node
            logits[:,:,0] = 1
        return logits, clone_mask

    def forward(self, encoder_context, targets):
        '''
        input: [batch_size, num_nodes (sequence_length), static_features]
        '''

        batch_size = targets.size(0)
        time_steps = targets.size(1)  # kanei tosa bhmata osa to sequence length poy dinetai

        decoder_input = torch.ones(batch_size, 1, hidden_size)

        # we calculate loss per batch here
        loss = 0
        tours = []
        chosen_indexes  = None
        mask = torch.zeros(batch_size, time_steps).byte()

        context = encoder_context
        for time_step in range(time_steps):
            assert decoder_input.size(0) == batch_size
            assert decoder_input.size(1) == 1 # sequence size is 1, takes 1 by 1 the input seq
            assert decoder_input.size(2) == hidden_size

            decoder_output,  context = self.decoder(decoder_input, context)


            # TODO: figure out pws pairnoume to actual result apo to decoder output?
            logits = self.decoderOutputToActualProbabilities(decoder_output) # [batch_size, seq_len ]

            ## TODO: add mask!
            masked_logits, mask = self.apply_mask_to_logits(logits, mask, chosen_indexes)


            softmax_logits = F.softmax(masked_logits, dim=2) # exei size [batch_size, 1]

            isAllNan = torch.all(softmax_logits.isnan())
            if isAllNan:
                break

            chosen_indexes = torch.argmax(softmax_logits, dim=2).float()

            loss += self.criterion(softmax_logits.squeeze(1), targets[:, time_step])

            tours.append(chosen_indexes.unsqueeze(1))

            # decoder input is next item in targets
            decoder_input = self.targetToDecoderInput(targets[:, time_step].unsqueeze(1).unsqueeze(1).float())
        loss /= batch_size
        tours = torch.cat(tours, 2)

        return loss, tours


class ClassicSeq2SeqTSPModel(nn.Module):
    def __init__(self,sequence_length):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(sequence_length)
    def forward(self, inputs, targets):
        encoded, encoderContext = self.encoder(inputs)
        loss, tours = self.decoder(encoderContext,  targets)
        return loss, tours

def trainClassicSeq2SeqTSP(train_dataset, test_dataset,epochs,experiment_details, batch_size = 10, num_nodes=13,lr=1e-4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    model = ClassicSeq2SeqTSPModel(sequence_length=num_nodes)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    val_loss = []
    train_loss= []
    val_loss_per_epoch = []
    losses_per_epoch = []

    tour_lengths_per_epoch = []

    for epoch in range(epochs):

        model.train()

        loss_at_epoch = 0.0
        val_loss_at_epoch = 0.0
        iterator = tqdm(train_loader, unit='Batch')
        # gia 1 item tou kathe batch
        average_tour_length = 0

        for batch_id, sample_batch in enumerate(iterator):
            train_batch = Variable(sample_batch['Points'])
            target_batch = Variable(sample_batch['Solution'])

            loss, chosen_routes = model(train_batch, target_batch)

            loss_at_epoch += loss.detach().item()
            optimizer.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1., norm_type=2)
            loss.backward()
            optimizer.step()


            train_loss.append(loss.data.item())


            for tour in range(batch_size):
                points = sample_batch['Points'][tour]
                distance_matrix_array = distance_matrix(points,points)
                content_from_my_model, tour_length = get_tour_length_from_distance_matrix(chosen_routes[tour].squeeze(0).long(), distance_matrix_array)
                average_tour_length += tour_length.item()



        # epoch finished
        model.eval()
        for val_batch in validation_loader:
            train_batch = Variable(val_batch['Points'])
            target_batch = Variable(val_batch['Solution'])
            loss, outputs = model(train_batch, target_batch)
            val_loss.append(loss.data.item())
            val_loss_at_epoch += loss.detach().item()
            # for this tour, what did the model find and what did the target was?
            format_tours(target_batch, outputs.squeeze(1).int())


        tour_lengths_per_epoch.append(average_tour_length/batch_size)

        print(f'Loss at epoch {epoch}: {loss_at_epoch}')
        losses_per_epoch.append(loss_at_epoch)
        val_loss_per_epoch.append(val_loss_at_epoch)

    # Training finished
    #experiment_details = f'epochs{epochs}_seqLen{num_nodes}_batch{batch_size}_lr{lr}'
    plot_train_and_validation_loss(epoch, losses_per_epoch, val_loss_per_epoch, experiment_details, "classicSeqToSeq")


    plot_average_tour_length(tour_lengths_per_epoch, experiment_details)

    return model,outputs