import dna_dataset

test_dna_dataset = dna_dataset.Dataset(100, 5, 5, 5)

print(test_dna_dataset.dna_sequence)

for i in range(len(test_dna_dataset)):
    print(test_dna_dataset[i])