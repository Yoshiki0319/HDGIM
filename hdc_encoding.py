import dna_dataset
import torch

class Encoder:
    def __init__(self, dimension, dna_sequence_length, dna_subsequences_length, number_of_true, number_of_false):
        self.dimension = dimension
        self.dna_sequence_length = dna_sequence_length
        self.dna_subsequences_length = dna_subsequences_length
        self.number_of_true = number_of_true
        self.number_of_false = number_of_false
        
        self.dna_dataset = dna_dataset.Dataset(dna_sequence_length, dna_subsequences_length, number_of_true, number_of_false)

        self.dna_sequence = self.dna_dataset.dna_sequence
        self.dna_subsequences = self.dna_dataset.samples
        
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
            first_base_num = subsequence[0].item()
            chunk_hypervector = self.base_hypervectors[first_base_num].unsqueeze(0)
            
            for base_num in subsequence[1:]:
                base_index = base_num.item()
                base_hypervector = self.base_hypervectors[base_index]
                chunk_hypervector = chunk_hypervector * base_hypervector.unsqueeze(0)  

            chunk_hypervector = torch.roll(chunk_hypervector, shifts=shift, dims=1).squeeze(0)
            chunk_hypervectors.append(chunk_hypervector)
        
        self.hdc_library = torch.stack(chunk_hypervectors)
        self.encoded_hypervector = torch.sum(self.hdc_library, dim=0)
        return self.encoded_hypervector

