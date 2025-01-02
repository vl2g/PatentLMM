# PatentLMM
Official Implementation of the PatentLMM: Large Multimodal Model for Generating Descriptions for Patent Figures (AAAI 2025) Paper

[Paper]() | [Project Page](https://vl2g.github.io/projects/PatentLMM/)

## Setting up the repo

1. Clone the repo
```
git clone https://github.com/vl2g/PatentLMM/
cd PatentLMM
```

2. Install the environment
```
conda env create -f patentlmm.yml
pip3 install -e .
```

### Download Checkpoints

The pre-trained checkpoints for PatentMME and PatentLMM are provided below:

| **PatentMME**| **PatentLMM-brief** | **PatentLMM-detailed** 
|------------------------------------------------------|-------------------------------------------------------|-------------------------------------------------------
| [Download](https://example.com/patentmme-large)      |  [Download](https://example.com/patentlmm-large)      | [Download](https://example.com/patentlmm-large)


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
