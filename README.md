# HDGIM: Hyperdimensional DNA Search

This project implements a **hyperdimensional computing (HDC) pipeline for DNA sequence search**, with hardware-aware modeling of quantization and noise.

## Overview

* Encode DNA sequences into high-dimensional vectors
* Apply quantization to simulate low-precision memory (FeFET-style)
* Inject noise to model device variability
* Perform similarity search using a Hamming-like distance

## Pipeline

1. **Dataset generation**
   Synthetic DNA sequences with matching / non-matching labels

2. **HDC encoding**

   * Base → random hypervector
   * Position → circular shift
   * Sequence → binding (multiplication)

3. **Quantization**

   * CDF-based mapping
   * Discretization into `2^bit_precision` levels

4. **Noise injection**

   * Random perturbation per dimension

5. **Similarity search**

   * L1 (Hamming-like distance)
     
## Installation

```bash
git clone https://github.com/Yoshiki0319/HDGIM.git
cd HDGIM
pip install -r requirements.txt
```

## Usage

```python
from hdgim import HDGIM

model = HDGIM(dimension=10000, bit_precision=3, noise=0.1)
model.train()
model.evaluate()
```

## Reference
```bash
@INPROCEEDINGS{10137331,
  author={Barkam, Hamza Errahmouni and Yun, Sanggeon and Genssler, Paul R. and Zou, Zhuowen and Liu, Che-Kai and Amrouch, Hussam and Imani, Mohsen},
  booktitle={2023 Design, Automation & Test in Europe Conference & Exhibition (DATE)}, 
  title={HDGIM: Hyperdimensional Genome Sequence Matching on Unreliable highly scaled FeFET}, 
  year={2023},
  volume={},
  number={},
  pages={1-6},
  keywords={Training;Adaptation models;Computational modeling;Genomics;Computer architecture;Iron;Robustness},
  doi={10.23919/DATE56975.2023.10137331}}
```
