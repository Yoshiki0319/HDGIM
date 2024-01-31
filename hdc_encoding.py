import dna_dataset
import dna_to_numbers
import torch

class Encoder:
    def __init__(self, dimension, dna_sequence_length, dna_subsequences_length, number_of_samples):
        self.dimension = dimension
        self.dna_sequence_length = dna_sequence_length
        self.dna_subsequences_length = dna_subsequences_length
        self.number_of_samples = number_of_samples
        
        self.dna_converter = dna_to_numbers.Encoder(dna_sequence_length, dna_subsequences_length)
        self.dna_dataset = dna_dataset.Dataset(dna_sequence_length, dna_subsequences_length, number_of_samples)

        self.dna_sequence = self.dna_converter.dna_num_sequence()
        self.dna_subsequences = self.dna_converter.generate_dna_subsequences()
        
        self.base_hypervectors = self.generate_base_hypervectors()

        self.encoded_hypervector = None
        self.hdc_library = None

    def generate_base_hypervectors(self):
        return {
            num: torch.empty(self.dimension).uniform_(-torch.pi, torch.pi)
            for num in range(4)
        }
    
    def binding(self):
        chunk_hypervectors = []
        for shift, subsequence in enumerate(self.dna_subsequences):
            chunk_hypervector = torch.ones(1, self.dimension)

            for base_num in subsequence:
                base_index = base_num.item() 
                base_hypervector = torch.roll(self.base_hypervectors[base_index], shifts=shift, dims=0)
                chunk_hypervector = torch.squeeze(torch.mul(chunk_hypervector, base_hypervector))
            
            chunk_hypervectors.append(chunk_hypervector)
            
        self.hdc_library = torch.stack(chunk_hypervectors)
        self.encoded_hypervector = torch.sum(self.hdc_library, dim=0)
        return self.encoded_hypervector

