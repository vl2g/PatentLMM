# PatentLMM
Official Implementation of our AAAI 2025 Paper

## Setting up the Environment
The `patentlmm.yml` provides a conda environment to run inference and experiments with PatentLMM. To install and setup the environment, please run:

```
conda env create -f patentlmm.yml
```

Next, please run `pip install -e .` from the root of this repo to install this folder as python package.

## Downloading Checkpoints and Data

### Checkpoints
The pre-trained checkpoints for PatentMME and PatentLMM are listed below:

| **Pre-training Objectives for PatentMME**                  | **Download (PatentMME)**                                  | **Download (PatentLMM)**                                   
|----------------------------------------------------|------------------------------------------------------|-------------------------------------------------------
| Masked LM + Layout-Aware MIM                | [Download](https://example.com/patentmme-base)       | [Download](https://example.com/patentlmm-base)        
|  Masked LM + Layout-Aware MIM + Patch Classification                      | [Download](https://example.com/patentmme-large)      |  [Download](https://example.com/patentlmm-large)       


### Data
We provide data in three parts:
- A json file with `image_ids` as keys, and its internet URL and corresponding brief and detailed description here. As mentioned in the paper, the descriptions in this file are clipped at 500 tokens. This should be used to download the images and for any custom pre-processing.
- A processed json file to work with llava-like framework. This is primarily used for training our model, along with the images downloaded using earlier json file.
- Three text files detailing the split of data into train, validation and test sets.

The download links are given below:
Compiled Data with URLs: []
LLaVA-compitable training files: []
Splits files: []


## Training PatentLMM


## Acknowledgements
- This work was supported by the Microsoft Academic Partnership Grant (MAPG) 2023.
- We heavily re-utilize the code from the [LLaVA repository](https://github.com/haotian-liu/LLaVA) for our experiments and would like to thank the authors for open-sourcing it!
