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
conda activate patentlmm
pip3 install -e .

# install flash-attention
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip3 install flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# remove the wheel
rm flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Downloading and preparing data
The PatentDesc-355k dataset is provided [here](https://drive.google.com/file/d/1PqLxhrqLa6m4_CwD_S0dvvTZDQJZdKY_/view?usp=drive_link) as a json file with `image_ids` as keys, and its internet URL and corresponding brief and detailed description here. Below is an example showing data format.

```
{
    "US11036663B2__2": {
        "image_url": "...",
        "brief_description": "...",
        "detailed_description": "..."
    },
    "US11336511B2__54": {
        "image_url": "...",
        "brief_description": "...",
        "detailed_description": "..."
    },
    .
    .
    .
}
```

As mentioned in the paper, the detailed descriptions in this file are clipped at 500 tokens.

Follow the steps below to download the dataset in appropriate format:

1.  
    ```
    mkdir DATASET
    cd DATASET
    ```

2.  Download [PatentDesc-355k.json](https://drive.google.com/file/d/1PqLxhrqLa6m4_CwD_S0dvvTZDQJZdKY_/view?usp=drive_link)

3.  
    ```
    mkdir images
    cd images
    ```
    Download the images using the given `image_url` from the json file. Please follow the naming convention of `image_id.png` for saving the images.
    ```
    cd ..
    ```

4.  Download the text files listing image_ids corresponding to train, val and test splits from [here](https://drive.google.com/drive/folders/12LXLU2lJtFdw4yev0E7MJnK1Suk-FL9U?usp=sharing).

5.  We utilize the LayoutLMv3 preprocessor which uses off-the-shelf Tesseract OCR engine, to extract OCR text from patent images. For convenience, we provide the json file with extracted OCR [here]().

6.  Run the following command to create data in LLaVA format for training/validation.
    ```
    mkdir llava_json
    cd ..
    python prep_patent_data.py --desc_type [brief/detailed] --split [train/val] --data_dir [path to DATASET directory]
    ```

Finally, the `DATASET` directory should have the following structure:
```
│DATASET│
│
├── PatentDesc-355k.json
├── ocr.json
├── splits
│   ├── all.txt
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── llava_json
│   ├── brief_train
│   ├── brief_val
│   ├── detailed_train
│   └── detailed_val
└── images
    ├── US11036663B2__2.png
    ├── US11336511B2__54.png
    .
    .
    . 
```

## Downloading Checkpoints

The pre-trained checkpoints for PatentMME, PatentLMM and PatentLLaMA are provided below:

| **PatentMME**| **PatentLMM-brief** | **PatentLMM-detailed** | **PatentLLaMA**
|------------------------------------------------------|-------------------------------------------------------|-------------------------------------------------------|-------------------------------------------------------
| [Download](https://drive.google.com/drive/folders/1n0kriDeXjnbw9hNVJ1FgdMogt5yHcD35?usp=sharing)      |  [Download](https://example.com/patentlmm-large)      | [Download](https://example.com/patentlmm-large)    |    [Download](https://example.com/patentllama)

Download and unzip the respective checkpoints in a `checkpoints` directory.

## Training PatentLMM
We follow two-stage strategy to train PatentLMM. To train the projection layer in stage-1, run:
```
bash scripts/v1_5/train_patentlmm_stage1.sh
```
To train for stage-2:
```
bash scripts/v1_5/train_patentlmm_stage2.sh
```

## Inference
To see the inference results with our trained model, run the command below:
```
python finetuned_inference.py --path_to_ckp [path to global_step/mp_rank_00_model_states.pt from checkpoint] --data_dir [path to DATASET directory] --desc_type [brief/detailed] --output_file [json file name where to save results]
```

## Cite us
- If you find this work useful for your research, please consider citing.
```
@inproceedings{shukla2025patentlmm,
  author    = "Shukla, Shreya and 
              Sharma, Nakul and 
              Gupta, Manish and
              Mishra, Anand",
  title     = "PatentLMM: Large Multimodal Model for Generating Descriptions for Patent Figures",
  booktitle = "AAAI",
  year      = "2025",
}
```

## Acknowledgements
- This work was supported by the Microsoft Academic Partnership Grant (MAPG) 2023.
- We would like to thank the authors of [LLaVA](https://github.com/haotian-liu/LLaVA), [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3) and [OCR-VQGAN](https://github.com/joanrod/ocr-vqgan) for open-sourcing their code and checkpoints!
