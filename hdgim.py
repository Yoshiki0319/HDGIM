import dna_dataset
import torch
from torch.distributions.normal import Normal
import random
from torch.utils.data import DataLoader

class HDGIM:
    def __init__(self, dimension, dna_sequence_length, dna_subsequences_length, number_of_true, number_of_false, bit_precision, noise, seed=42):
        self.dimension = dimension
        self.dna_sequence_length = dna_sequence_length
        self.dna_subsequences_length = dna_subsequences_length
        self.number_of_true = number_of_true
        self.number_of_false = number_of_false
        self.bit_precision = bit_precision
        self.noise = noise
        random.seed(seed)
        
        self.dna_dataset = dna_dataset.Dataset(dna_sequence_length, dna_subsequences_length, number_of_true, number_of_false)

        self.dna_sequence = self.dna_dataset.dna_sequence
        self.dna_subsequences = self.dna_dataset.samples
        
        self.base_hypervectors = self.generate_base_hypervectors()

        self.encoded_hypervector = None
        self.hdc_library = None

        self.quantized_hypervector_with_noise = None


    def generate_base_hypervectors(self):
        return {
            num: torch.empty(self.dimension).uniform_(-torch.pi, torch.pi)
            for num in range(4)
        }
    
    def binding(self):
        chunk_hypervectors = []
        for shift, subsequence in enumerate(self.dna_subsequences):
            chunk_hypervector = torch.ones((1, self.dimension))
            
            for base_num in subsequence:
                base_index = base_num.item()
                base_hypervector = self.base_hypervectors[base_index]
                chunk_hypervector = chunk_hypervector * base_hypervector  

            chunk_hypervector = torch.roll(chunk_hypervector, shifts=shift, dims=0) 
            chunk_hypervectors.append(chunk_hypervector)
        
        self.hdc_library = torch.stack(chunk_hypervectors)
        self.encoded_hypervector = torch.sum(self.hdc_library, dim=0)

    def binding_arbitrary_sequence(self, data):
        encoded_data_hypervector = torch.ones((1, self.dimension))
        
        for base_num in data:
            base_index = base_num.item()
            base_hypervector = self.base_hypervectors[base_index]
            encoded_data_hypervector = torch.mul(encoded_data_hypervector, base_hypervector)  
        
        return encoded_data_hypervector
    
    def quantize(self):
        mean = torch.mean(self.encoded_hypervector)
        std = torch.std(self.encoded_hypervector)
        normalized_hypervector = (self.encoded_hypervector - mean) / std

        normal_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        cdf_values = normal_dist.cdf(normalized_hypervector)

        binary_width = 1.0/(2**self.bit_precision)
        quantized_values = torch.floor(cdf_values / binary_width)

        self.quantized_hypervector = quantized_values
    
    def quantize_arbitrary_sequence(self, data):
        mean = torch.mean(data)
        std = torch.std(data)
        normalized_data = (data - mean) / std

        normal_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        cdf_values = normal_dist.cdf(normalized_data)

        binary_width = 1.0/(2**self.bit_precision)
        quantized_values = torch.floor(cdf_values / binary_width)

        return quantized_values
    
    def adding_noise(self):
        self.quantized_hypervector_with_noise = self.quantized_hypervector
        for i, value in enumerate(self.quantized_hypervector):
            is_changed = random.random() < self.noise
            if not is_changed:
                continue

            is_in_range = 0
            num = value.item()

            if num == 0:
                is_in_range = 1
            elif num == pow(2, self.bit_precision):
                is_in_range = 0
            else:
                is_in_range = random.randint(0, 1)

            changed_value = -1 if is_in_range == 0 else 1
            noise_value = num + changed_value
            self.quantized_hypervector_with_noise[i] = noise_value
    
    def hamming_distance(self, hypervector1, hypervector2):
        return torch.sum(torch.abs(hypervector1 - hypervector2))
    
    def train(self, epoch, lr, threshold, return_info, return_data):
        train_dataset = self.dna_dataset
        
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        list_accuracy = []
        true_similarity = []
        false_similarity = []

        if return_info:
            print("Train size: {}".format(len(train_dataset)))

        for e in range(epoch):
            correct_cnt = 0
            tp_cnt = 0
            tn_cnt = 0
            fp_cnt = 0
            fn_cnt = 0

            true_similarity.append([])
            false_similarity.append([])

            for data in train_dataloader:
                sim = 0

                query = torch.squeeze(data['dna_subsequence'])
                encoded_query = self.binding_arbitrary_sequence(query)
                quantized_query = self.quantize_arbitrary_sequence(encoded_query)

                label = data['label'].item()
                sim = (self.hamming_distance(self.quantized_hypervector_with_noise, quantized_query)) / self.dimension
                print(sim)

                if (sim < threshold) and not label:
                    tn_cnt += 1
                    correct_cnt += 1
                elif (sim >= threshold) and label:
                    tp_cnt += 1
                    correct_cnt += 1
                elif (sim >= threshold) and not label:
                    self.encoded_hypervector -= lr * encoded_query
                    self.quantize()
                    self.adding_noise()
                    fn_cnt += 1
                elif (sim < threshold) and label:
                    self.encoded_hypervector += lr * encoded_query
                    self.quantize()
                    self.adding_noise()
                    fp_cnt += 1

                if label:
                    true_similarity[e].append(sim)
                else:
                    false_similarity[e].append(sim)

            accuracy = round(correct_cnt * 100 / len(train_dataset), 2)
            list_accuracy.append(accuracy)

            if return_info:
                print(f"Epoch: {e} | Accuracy: {accuracy}% | TP: {tp_cnt} | TN: {tn_cnt} | FP: {fp_cnt} | FN: {fn_cnt}")

        if return_data:
            return list_accuracy, true_similarity, false_similarity