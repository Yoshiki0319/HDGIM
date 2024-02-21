import dna_gpu
import random
import torch
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dna_sequence_length, dna_subsequence_length, number_of_true, number_of_false, seed=42, device='cuda'):
        self.dna_sequence_length = dna_sequence_length
        self.dna_subsequence_length = dna_subsequence_length
        self.number_of_true = number_of_true
        self.number_of_false = number_of_false
        self.seed = seed
        random.seed(self.seed)
        self.device = device

        # Generate a DNA sequence
        dna_sequence_generator = dna_gpu.DNASequenceGenerator(dna_sequence_length)
        self.dna_sequence = dna_sequence_generator.generate_dna_sequence_tensor()

        # Generate true subsequences
        self.encoder = dna_gpu.Encoder(dna_sequence_length, dna_subsequence_length)
        self.dna_true_subsequences = self.encoder.generate_dna_subsequences()

        # Generate false subsequences
        self.dna_false_subsequences = []
        false_subsequences_set = set()  # Set for checking uniqueness of tuples

        while len(self.dna_false_subsequences) < self.number_of_false:
            false_subsequence = [random.choice([0, 1, 2, 3]) for _ in range(self.dna_subsequence_length)]
            false_subsequence_tuple = tuple(false_subsequence)  # Convert list to tuple

            if false_subsequence_tuple not in false_subsequences_set:
                # Check if the subsequence is not included in the DNA sequence
                if not self.encoder.is_included(torch.tensor(false_subsequence, dtype=torch.long, device=self.device)):
                    false_subsequences_set.add(false_subsequence_tuple)  # Add tuple to set
                    self.dna_false_subsequences.append(false_subsequence)

        # Convert the list of unique subsequences to a tensor
        self.dna_false_subsequences = torch.stack([torch.tensor(subseq, dtype=torch.long) for subseq in self.dna_false_subsequences]).to(self.device)


        # Combine true and false subsequences
        # use index_list to shuffle dna_true_subsequences
        numbers = list(range(0, len(self.dna_true_subsequences)))
        index_tensor = torch.tensor(random.sample(numbers, number_of_true), dtype=torch.long, device=self.device)
        true_subsequences_subset = torch.index_select(self.dna_true_subsequences, 0, index_tensor)
        self.samples = torch.cat((true_subsequences_subset, self.dna_false_subsequences[:number_of_false]), dim=0)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        label = self.encoder.is_included(self.samples[idx])
        dna_subsequence = self.samples[idx]
        return {'label': label, 'dna_subsequence': dna_subsequence}
    
