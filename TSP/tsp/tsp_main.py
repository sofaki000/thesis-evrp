from TSP.tsp.models.classicSeq2seqModel import trainClassicSeq2SeqTSP
from TSP.tsp.tsp_train_model import train_tsp_model
from datasets.TSP_dataset import TSPDataset
from or_tools_comparisons.common_utilities import print_solution, get_tour_length_from_distance_matrix

if __name__ == '__main__':

    epochs  = 5
    num_nodes = 5
    train_size = 40
    test_size = 10
    batch_size = 5

    train_dataset = TSPDataset(train_size, num_nodes)

    test_dataset = TSPDataset(test_size, num_nodes)

    my_model, outputs = trainClassicSeq2SeqTSP(train_dataset,
                                        test_dataset,
                                        epochs,
                                        batch_size ,
                                        num_nodes )

