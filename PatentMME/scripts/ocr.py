from transformers import AutoProcessor
import argparse
import tqdm
import os
from PIL import Image
import json
from math import ceil


def chunk_into_n(lst, n):
    size = ceil(len(lst) / n)
    return list(
        map(lambda x: lst[x * size:x * size + size],
        list(range(n)))
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hupd_imgs_dir", default='/DATA/nakul/PatentQA/gen_data/DATASET/images_combined/')
    parser.add_argument("--split", required=True)
    parser.add_argument("--image_size", default=384)
    parser.add_argument("--output_dir", default='/DATA/nakul/PatentQA/gen_data/DATASET/LayoutLM_ocr/', help='path to the directory where word-level OCR are to be saved')

    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-large", apply_ocr=True)
    processor.image_processor.size = {'height': args.image_size, 'width': args.image_size}

    im_paths = sorted([os.path.join(args.hupd_imgs_dir, im) for im in os.listdir(args.hupd_imgs_dir)])
    im_paths = chunk_into_n(im_paths, 32)[int(args.split)]

    HUPD_OCR = {}
    os.makedirs(args.output_dir, exist_ok=True)

    for im_pth in tqdm.tqdm(im_paths):
        HUPD_OCR[im_pth.split('/')[-1]] = {}
        img = Image.open(im_pth).convert('RGB')
        
        encoding = processor(img, padding='longest')
        
        HUPD_OCR[im_pth.split('/')[-1]]['input_ids'] = encoding['input_ids']
        HUPD_OCR[im_pth.split('/')[-1]]['attention_mask'] = encoding['attention_mask']
        HUPD_OCR[im_pth.split('/')[-1]]['bbox'] = encoding['bbox']

    with open(os.path.join(args.output_dir, f'hupd_ocr_{args.split}.json'), 'w') as f:
        json.dump(HUPD_OCR, f, indent=2)
