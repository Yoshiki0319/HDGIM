import random
import torch

class DNASequenceGenerator:
    def __init__(self, dna_sequence_length, seed=42):
        self.dna_sequence_length = dna_sequence_length
        self.seed = seed
        self.bases = ['A', 'T', 'C', 'G']
        random.seed(self.seed)
    
    def generate_dna_sequence_tensor(self):
        # The original mapping of nucleotide bases to integers
        convert = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        
        # Generate a random DNA sequence with an additional step to shuffle the sequence to ensure more randomness
        dna_sequence = [random.choice(self.bases) for _ in range(self.dna_sequence_length)]
        random.shuffle(dna_sequence)  # Shuffle the generated sequence to add more randomness
        
        # Convert the sequence to a tensor of integers
        return torch.tensor([convert[base] for base in dna_sequence], dtype=torch.long)

class Encoder:
    def __init__(self, dna_sequence_length, dna_subsequences_length):
        self.dna_subsequences_length = dna_subsequences_length
        generator = DNASequenceGenerator(dna_sequence_length)
        self.dna_sequence_tensor = generator.generate_dna_sequence_tensor()
        self.dna_true_subsequences = self.generate_dna_subsequences()
    
    def generate_dna_subsequences(self):
        dna_true_subsequences = []
        for i in range(len(self.dna_sequence_tensor) - self.dna_subsequences_length + 1):
            subsequence = self.dna_sequence_tensor[i:i + self.dna_subsequences_length]
            dna_true_subsequences.append(subsequence)
        return torch.stack(dna_true_subsequences)

    def is_included(self, dna_subsequence_tensor):
        length = dna_subsequence_tensor.size(0)
        for i in range(len(self.dna_sequence_tensor) - length + 1):
            if torch.equal(self.dna_sequence_tensor[i:i + length], dna_subsequence_tensor):
                return True
        return False
