from TSP.tsp.models.classicSeq2seqModel import trainClassicSeq2SeqTSP
from TSP.tsp.tsp_train_model import train_tsp_model
from datasets.TSP_dataset import TSPDataset
from or_tools_comparisons.common_utilities import print_solution, get_tour_length_from_distance_matrix

if __name__ == '__main__':

    epochs  = 15
    num_nodes = 5
    train_size = 400
    test_size = 10
    batch_size = 5

    train_dataset = TSPDataset(train_size, num_nodes)

    test_dataset = TSPDataset(test_size, num_nodes)
    experiment_details = f'supervised_epochs{epochs}_seqLen{num_nodes}_batch{batch_size}_lr{lr}'

    my_model, outputs = trainClassicSeq2SeqTSP(train_dataset,
                                        test_dataset,
                                        epochs,
                                               experiment_details,
                                        batch_size ,
                                        num_nodes,
                                        lr=1e-2)

