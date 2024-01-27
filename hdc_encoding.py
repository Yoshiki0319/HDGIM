import dna_dataset
import dna_to_numbers
import torch

class Encoder:
    def __init__(self, dimention, dna_sequence_length, dna_queries_length, number_of_samples):
        self.dimention = dimention
        self.dna_sequence_length = dna_sequence_length
        self.dna_queries_length = dna_queries_length
        self.number_of_samples = number_of_samples
        
        self.d_to_n = dna_to_numbers.Encoder(self.dna_sequence_length, self.dna_queries_length)
        self.DNA_dataset = dna_dataset.Dataset(self.dna_sequence_length, self.dna_queries_length, number_of_samples)

        self.dna_sequence = None
        self.dna_queries = None
        self.hyper_dimensional_vector = None

    def generate_dna_sequence(self):
        self.dna_sequence = self.d_to_n.dna_num_sequence()
    
    def generate_dna_queries(self):
        self.dna_queries = self.d_to_n.generate_dna_queries()
    
    def generate_hyper_dimensional_vector(self, input_tensor):
        return ((input_tensor.float() / 3) * 2 * torch.pi) - torch.pi
        
    

    

    

    


