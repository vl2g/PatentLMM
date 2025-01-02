from dataset import PatentDescDataset, custom_collate
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from llava import *
import torch
import argparse
import json
import tqdm
from peft import LoraConfig, get_peft_model
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="liuhaotian/llava-v1.5-7b")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default='microsoft/layoutlmv3-large')
    mm_vision_select_layer: Optional[int] = field(default=-2)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_vision_select_feature: Optional[str] = field(default="patch")

def tokenizer_image_token(prompt, tokenizer, image_token_index=-200, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    ##### change here for detailed description
    # line47:  return torch.tensor(input_ids, dtype=torch.long)
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids[:1024], dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for processing (default: 4)")
    parser.add_argument("--path_to_ckp", type=str, required=True, help="Path to global_step/mp_rank_00_model_states.pt from checkpoint")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--desc_type", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parser.parse_args()

    test_ds = PatentDescDataset('test', desc_type=args.desc_type, data_dir=args.data_dir, data_len=-1)

    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=custom_collate)
                
    tokenizer = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", fast=False)
    
    model = LlavaLlamaForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b")

    lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=find_all_linear_names(model),
            lora_dropout=0.,
            bias="none",
            task_type="CAUSAL_LM",
        )
    # model.to(torch.bfloat16)

    model_args = ModelArguments()

    model = get_peft_model(model, lora_config)
    model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=None
        )

    vision_tower = model.get_vision_tower()
    # vision_tower.to(torch.bfloat16)

    model.config.mm_use_im_start_end = False
    model.config.mm_projector_lr = 2e-5
    model.config.mm_use_im_patch_token = False
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    model.config.image_aspect_ratio = 'pad'
    model.config.tokenizer_padding_side = 'right'
    model.config.tokenizer_model_max_length = 2048

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(args.path_to_ckp)['module'])
    model = model.to(device)
    model.eval()

    processor = vision_tower.processor

    prompt = f"<image>\nUSER: Write a {args.desc_type} description for this patent image.\nASSISTANT:"

    generated_descriptions = {}
    
    for batch in tqdm.tqdm(test_dl):
        inputs = processor(batch['img'], return_tensors='pt')
        
        inputs['images'] = inputs.pop('pixel_values').to(device)
        inputs['input_ids'] = torch.stack([tokenizer_image_token(p, tokenizer, return_tensors='pt') for p in [prompt]*args.batch_size], dim=0).to(device)
        inputs['ocr_input_ids'] = batch['ocr_input_ids'].to(device)
        inputs['ocr_attention_mask'] = batch['ocr_attention_mask'].to(device)
        inputs['ocr_bboxes'] = batch['ocr_bboxes'].to(device)

        description_ids = model.generate(**inputs, max_length=512)
        description_ids[description_ids == -200] = 1
        description_text = tokenizer.batch_decode(description_ids, clean_up_tokenization_spaces=False)
        
        # print(description_text)
        for i in range(len(batch['fig_id'])):
            generated_descriptions[batch['fig_id'][i]] = description_text[i].split('image.\nASSISTANT:')[-1]

        with open(args.output_file, "w") as outfile:
            json.dump(generated_descriptions, outfile, indent=2)
