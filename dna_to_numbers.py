import dna
import torch
import numpy as np
from itertools import product

class Encoder:
    def __init__(self, dna_sequence_length, dna_queries_length):
        self.dna_sequence_length = dna_sequence_length
        self.dna_queries_length = dna_queries_length
        
        self.dna_generate = dna.random_generate(self.dna_sequence_length, self.dna_queries_length)
        self.convert = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        self.dna_sequence = [self.convert[base] for base in self.dna_generate.generate_dna_sequence()]
        self.dna_query = [self.convert[base] for base in self.dna_generate.generate_dna_query()]
        
        self.dna_patterns = None

    def hd_dna_sequence(self):
        return torch.tensor(self.dna_sequence)
    
    def hd_dna_query(self):
        return torch.tensor(self.dna_query)
    
    def generate_all_patterns(self):
        pattern = torch.arange(0, 4)
        self.dna_patterns = torch.tensor(list(product(*[pattern]*self.dna_queries_length)))
        return self.dna_patterns

    def generate_true_dna_queries(self):
        unique_dna_queries = []

        for i in range(self.dna_sequence_length - self.dna_queries_length + 1):
            subsequence_tensor = torch.tensor(self.dna_sequence[i:i+self.dna_queries_length], dtype=torch.int64)
            if subsequence_tensor.tolist() not in unique_dna_queries:
                unique_dna_queries.append(subsequence_tensor.tolist())
        
        return torch.tensor(unique_dna_queries, dtype=torch.int64)