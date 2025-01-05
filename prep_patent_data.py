from dataset import PatentDescDataset
import json
from tqdm import tqdm
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc_type', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    print(f"Preparing {args.desc_type} descriptions for {args.split} split")
    llava_dataset = []

    data = PatentDescDataset(split=args.split, desc_type=args.desc_type, data_dir=args.data_dir, ocr_only=True)

    spcl_tokens = ["<image>", "<im_patch>", "<im_start>", "<im_end>", "<image-placeholder>", "<", ">"]
    reps = {x: x.replace('<', '[[').replace('>', ']]') for x in spcl_tokens}

    for sample in tqdm(data):
        desc = sample['description']

        for tok in spcl_tokens:
            desc = desc.replace(tok, reps[tok])

        _sample = {
            "id": sample['fig_id'],
            "image": f"{sample['fig_id']}.png",
            "conversations": [
            {
                "from": "human",
                "value": f"<image>\nWrite a {args.desc_type} description for this patent image."
            },
            {
                "from": "gpt",
                "value": desc
            },
            ]
        }

        llava_dataset.append(_sample)

    print(f"Writing {args.desc_type} descriptions for {args.split} split")

    with open(os.path.join(args.data_dir, f'llava_json/{args.desc_type}_{args.split}'), 'w') as f:
        json.dump(llava_dataset, f)
