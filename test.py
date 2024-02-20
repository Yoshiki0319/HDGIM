import numpy as np
import hdgim
import torch

dimension = 512
dna_sequence_length = 1000
dna_subsequences_length = 20
number_of_true = 50
number_of_false = 50
bit_precision = 2
noise = 0.5

hdgim_instance = hdgim.HDGIM(dimension, dna_sequence_length, dna_subsequences_length, number_of_true, number_of_false, bit_precision, noise)

hdgim_instance.binding()
hdgim_instance.quantize()
hdgim_instance.adding_noise()

epoch = 30
lr = 0.1
thresholds = np.arange(1, 5, 0.01) 

best_accuracy = 0
best_threshold = None

for threshold in thresholds:
    print("Threshold: {}".format(threshold))
    hdgim_instance.train(epoch=epoch, lr=lr, threshold=threshold, return_info=True, return_data=True)
