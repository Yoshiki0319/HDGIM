import hdc
import random
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dna_sequence_length, dna_queries_length, number_of_queries):
        self.dna_sequence_length = dna_sequence_length
        self.dna_queries_length = dna_queries_length
        self.number_of_queries = number_of_queries
        
        self.hdc = hdc.Encoder(self.dna_sequence_length, self.dna_queries_length)
        self.dna_sequence = self.hdc.hd_dna_sequence()
        self.dna_true_queries = self.hdc.generate_true_dna_queries()
        self.dna_false_queries = set(map(tuple, self.hdc.generate_all_patterns().numpy())) - set(map(tuple, self.dna_true_queries.numpy()))

        # Convert sets to lists to enable indexing
        self.dna_true_queries = list(self.dna_true_queries)
        self.dna_false_queries = list(self.dna_false_queries)
        
        # Keeping as sets if random access is sufficient
        # Convert to list only if sequential access or specific indexing is required

        # Create a list of indices with equal numbers of true and false queries
        self.query_indices = ['true'] * number_of_queries + ['false'] * number_of_queries
        random.shuffle(self.query_indices)

    def __len__(self):
        return 2 * self.number_of_queries
    
    def __getitem__(self, index):
        query_type = self.query_indices[index]
        if query_type == 'true':
            query_index = index % len(self.dna_true_queries)
            query = random.choice(self.dna_true_queries)  # Using random.choice for efficiency
            return {'label': True, 'dna_subsequence': query}
        else:
            query_index = index % len(self.dna_false_queries)
            query = random.choice(self.dna_false_queries)  # Using random.choice for efficiency
            return {'label': False, 'dna_subsequence': query}
