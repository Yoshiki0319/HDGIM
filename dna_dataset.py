import hdc
import random
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dna_sequence_length, dna_subsequences_length, number_of_queries):
        self.dna_sequence_length=dna_sequence_length
        self.dna_subsequences_length=dna_subsequences_length
        self.number_of_queries=number_of_queries
        
        self.hdc = hdc.Encoder(self.dna_sequence_length, self.dna_subsequences_length, self.number_of_queries)
        self.dna_sequence = self.hdc.generate_dna_sequence()
        self.dna_true_subsequences = self.hdc.generate_true_dna_subsequences()
        self.dna_false_subsequences = []

        for i, j in zip(self.dna_sequence, self.dna_true_subsequences):
            false_subseq = torch.masked_select(i, i!=j).unsqueeze(0)
            if false_subseq.size(1) < self.dna_subsequences_length:
                false_subseq = torch.cat((false_subseq, torch.zeros(1, self.dna_subsequences_length - false_subseq.size(1), dtype=torch.int64)), dim=1)
            self.dna_false_subsequences.append(false_subseq)
        
        true_size = self.dna_true_subsequences.size(1)
        self.dna_reference_library = torch.cat(
            (self.dna_true_subsequences, *self.dna_false_subsequences),
            dim=0
        )

    def __len__(self):
        return self.number_of_queries
    
    def __getitem__(self, index):
        if any(torch.equal(self.dna_true_subsequences, sub_array) for sub_array in self.dna_reference_library):
            result = {'label': True, 'dna_subsequence': self.dna_true_subsequences[index]}
        else:
            result = {'label': False, 'dna_subsequence': self.dna_false_subsequences[index]}
        return result
    
