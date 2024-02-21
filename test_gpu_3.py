import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdgim_gpu as hdgim

# パラメータ範囲の設定
dimensions = [512, 4500, 10000]
bit_precisions = [2, 3, 4]
noises = [0.2, 0.5, 0.8]
dna_sequence_length = 1000
dna_subsequences_length = 20
number_of_true = 50
number_of_false = 50
epoch = 10
lr = 1
thresholds = np.arange(2, 4, 0.001)

# 結果を保存するためのデータ構造
results = np.zeros((len(noises), len(dimensions), len(bit_precisions)))

for i, dimension in enumerate(dimensions):
    for j, bit_precision in enumerate(bit_precisions):
        for k, noise in enumerate(noises):
            hdgim_instance = hdgim.HDGIM(dimension, dna_sequence_length, dna_subsequences_length, number_of_true, number_of_false, bit_precision, noise)
            hdgim_instance.binding()
            hdgim_instance.quantize()
            hdgim_instance.adding_noise()

            best_accuracy = 0
            best_threshold = None
            consecutive_declines = 0  # 連続して精度が落ちる回数を追跡

            for threshold in thresholds:
                accuracy_list, true_sim, false_sim = hdgim_instance.train(epoch=epoch, lr=lr, threshold=threshold, return_info=True, return_data=True)
                current_best_accuracy = max(accuracy_list)
                
                if current_best_accuracy >= best_accuracy:
                    best_accuracy = current_best_accuracy
                    best_threshold = threshold
                    consecutive_declines = 0  # 精度が改善されたらリセット
                else:
                    consecutive_declines += 1
                
                if consecutive_declines >= 20:
                    break

            results[k, i, j] = best_accuracy
            print(f"Dimension: {dimension}, Bit Precision: {bit_precision}, Noise: {noise}, Best Accuracy: {best_accuracy}%, Best Threshold: {best_threshold}")

# ヒートマップの表示
for j, bit_precision in enumerate(bit_precisions):
    plt.figure(figsize=(10, 8))
    sns.heatmap(results[:, :, j], annot=True, fmt=".2f", xticklabels=dimensions, yticklabels=noises, cmap="viridis")
    plt.title(f'Heatmap of Best Accuracy for Bit Precision: {bit_precision}')
    plt.xlabel('Dimension')
    plt.ylabel('Noise')
    plt.show()
