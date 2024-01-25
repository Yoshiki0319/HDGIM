import torch
import numpy as np

class Encoder:
    def __init__(self, dna_sequence_length, dna_subsequences_length, number_of_queries):
        self.dna_sequence_length = dna_sequence_length
        self.dna_subsequences_length = dna_subsequences_length
        self.number_of_queries = number_of_queries
        self.dna_sequence = torch.randint(0, 4, (self.dna_sequence_length,))

    def generate_dna_sequence(self):
        return self.dna_sequence
    
    def generate_dna_subsequences(self):
        value_range = np.arange(0, 4)
        dna_subsequences = np.array(np.meshgrid(*[value_range]*4)).T.reshape(-1, 4)
        return torch.tensor(dna_subsequences)

    def generate_true_dna_subsequences(self):
        true_dna_subsequences = []
        for i in range(self.dna_sequence_length - self.dna_subsequences_length + 1):
            subsequence_tensor = torch.tensor(self.dna_sequence[i:i+self.dna_subsequences_length], dtype=torch.int64)
            true_dna_subsequences.append(subsequence_tensor)
        return torch.stack(true_dna_subsequences)

