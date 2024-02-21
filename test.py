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

epoch = 10
lr = 1000
thresholds = np.arange(-1.0, 1.0, 0.001)

best_accuracy = 0
best_threshold = None

for threshold in thresholds:
    accuracy_list, true_sim, false_sim = hdgim_instance.training_full_precision(epoch=epoch, lr=lr, threshold=threshold, return_info=False, return_data=True)
    current_best_accuracy = max(accuracy_list)
    if current_best_accuracy > best_accuracy:
        best_accuracy = current_best_accuracy
        best_threshold = threshold
    print(f"Accuracy: {current_best_accuracy}% at Threshold: {threshold}")

# 最適な閾値と精度の表示
print(f"Best Accuracy: {best_accuracy}% at Best Threshold: {best_threshold}")



    
