import random
import torch
import numpy as np
from itertools import product

class Encoder:
    def __init__(self, dna_sequence_length, dna_queries_length):
        self.bases = ['A', 'T', 'C', 'G']
        self.convert = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

        self.dna_sequence_length = dna_sequence_length
        self.dna_queries_length = dna_queries_length

        self.dna_sequence = self.load_dna_sequence_from_file('dna_sequence.txt')
        
        self.dna_patterns = None
    
    def load_dna_sequence_from_file(self, filename):
        with open(filename, 'r') as file:
            sequence_str = file.readline().strip()
            sequence_list = list(sequence_str)
        return [self.convert[base] for base in sequence_list]
    
    def dna_num_sequence(self):
        return torch.tensor(self.dna_sequence)
    
    def generate_all_patterns(self):
        pattern = torch.arange(0, 4)
        self.dna_patterns = torch.tensor(list(product(*[pattern]*self.dna_queries_length)))
        return self.dna_patterns

    def generate_dna_queries(self):
        unique_dna_queries = []

        for i in range(self.dna_sequence_length - self.dna_queries_length + 1):
            subsequence_tensor = torch.tensor(self.dna_sequence[i:i+self.dna_queries_length], dtype=torch.int64)
            if subsequence_tensor.tolist() not in unique_dna_queries:
                unique_dna_queries.append(subsequence_tensor.tolist())
        
        return torch.tensor(unique_dna_queries, dtype=torch.int64)
    
    def get_dna_queries_as_dna_expression(self):
        dna_queries = self.generate_dna_queries()
        dna_queries_as_dna_expression = []

        for query in dna_queries:
            dna_expression = []
            for base in query:
                dna_expression.append(self.bases[base])
            dna_queries_as_dna_expression.append(dna_expression)

        return dna_queries_as_dna_expression