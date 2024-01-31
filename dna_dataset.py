import dna_to_numbers
import random
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dna_sequence_length, dna_subsequences_length, number_of_samples):
        self.dna_sequence_length = dna_sequence_length
        self.dna_subsequences_length = dna_subsequences_length
        self.number_of_samples = number_of_samples
        
        self.d_to_n = dna_to_numbers.Encoder(self.dna_sequence_length, self.dna_subsequences_length)
        self.dna_sequence = self.d_to_n.dna_num_sequence()
        self.dna_true_subsequences = self.d_to_n.generate_dna_subsequences()
        self.dna_false_subsequences = set(map(tuple, self.d_to_n.generate_all_patterns().numpy())) - set(map(tuple, self.dna_true_subsequences.numpy()))

        # Convert sets to lists to enable indexing
        self.dna_true_subsequences = list(self.dna_true_subsequences)
        self.dna_false_subsequences = torch.tensor(list(self.dna_false_subsequences))

        # Create a list of indices with equal numbers of true and false subsequences
        self.query_indices = ['true'] * number_of_samples + ['false'] * number_of_samples
        random.shuffle(self.query_indices)

        self.true_samples = None
        self.false_samples = None

    def __len__(self):
        return 2 * self.number_of_samples
    
    def __getitem__(self, index):
        query_type = self.query_indices[index]
        if query_type == 'true':
            query_index = index % len(self.dna_true_subsequences)
            query = random.choice(self.dna_true_subsequences)  # Using random.choice for efficiency
            self.true_samples = query
            return {'label': True, 'dna_query': query}
        else:
            query_index = index % len(self.dna_false_subsequences)
            query = random.choice(self.dna_false_subsequences)  # Using random.choice for efficiency
            self.false_samples = query
            return {'label': False, 'dna_query': query}
    
    def get_dna_sequence(self):
        return self.dna_sequence
    
    def get_dna_subsequences(self):
        return self.true_samples
    
    def get_false_dna_subsequences(self):
        return self.false_samples
