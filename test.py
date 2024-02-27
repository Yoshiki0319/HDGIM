import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdgim

dimensions = [512, 4000, 10000]
bit_precision = 0
noise = 0

dna_sequence_length = 1000
dna_subsequences_length = 20
number_of_true = 50
number_of_false = 50

epoch = 30
lr = 0.1
thresholds = np.arange(0.7, 1, 0.0001)

results = np.zeros(len(dimensions))

for i, dimension in enumerate(dimensions):
    hdgim_instance = hdgim.HDGIM(dimension, dna_sequence_length, dna_subsequences_length, number_of_true, number_of_false, bit_precision, noise)
    hdgim_instance.binding()

    best_accuracy = 0
    best_threshold = None

    for threshold in thresholds:
        print(f"Threshold: {threshold}")
        accuracy_list, true_sim, false_sim = hdgim_instance.training_full_precision(epoch=epoch, lr=lr, threshold=threshold, return_info=True, return_data=True)
        current_best_accuracy = max(accuracy_list)
                
        if current_best_accuracy >= best_accuracy:
            best_accuracy = current_best_accuracy
            best_threshold = threshold
                
        if best_accuracy == 100:
            break

    results[i] = best_accuracy
    print(f"Dimension: {dimension}, Best Accuracy: {best_accuracy}%, Best Threshold: {best_threshold}")

plt.figure(figsize=(10, 6))
plt.plot(dimensions, results, marker='o', linestyle='-', color='b')
plt.title('Best Accuracy vs. Dimension')
plt.xlabel('Dimension')
plt.ylabel('Best Accuracy (%)')
plt.ylim(0, 100)
plt.grid(True)
plt.show()

