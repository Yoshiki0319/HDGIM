import random

class random_generate:
    def __init__(self, dna_sequence_length, dna_query_length):
        self.dna_sequence_length = dna_sequence_length
        self.dna_query_length = dna_query_length
        self.bases = ['A', 'T', 'C', 'G']
    
    def generate_dna_sequence(self):
        return [random.choice(self.bases) for _ in range(self.dna_sequence_length)]
    
    def generate_dna_query(self):
        return [random.choice(self.bases) for _ in range(self.dna_query_length)]
