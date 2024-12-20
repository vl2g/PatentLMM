from PIL import Image
import argparse
import json
import os
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True, help='path to HUPD images directory')
    parser.add_argument('--frcnn_anns_file', required=True, help='path to frcnn generated annotations file')
    parser.add_argument("--input_size", type=int, default=384, help='processed image dimension')
    parser.add_argument("--output_dir", required=True, help='path to directory where new scaled annotations are to be saved')

    args = parser.parse_args()

    with open(args.frcnn_anns_file, 'r') as f:
        annotations = json.load(f)
        
    for i, img_id in enumerate(tqdm.tqdm(annotations)):
        img = Image.open(os.path.join(args.images_dir, img_id))
        orig_width, orig_height = img.size
        ls_anns = annotations[img_id]
        bboxes = []
        for i, anns in enumerate(ls_anns):
            anns['bbox'][0][0] = (anns['bbox'][0][0]/orig_width)*args.input_size
            anns['bbox'][0][1] = (anns['bbox'][0][1]/orig_height)*args.input_size
            anns['bbox'][0][2] = (anns['bbox'][0][2]/orig_width)*args.input_size
            anns['bbox'][0][3] = (anns['bbox'][0][3]/orig_height)*args.input_size
            anns['bbox'] = anns['bbox'][0]
            bboxes.append(anns['bbox'])
            ls_anns[i] = anns
        
        annotations[img_id] = ls_anns
        
        if i % 500 == 0:
            with open(os.path.join(args.output_dir, f'scaled_annotations.json'), 'w') as f:
                json.dump(annotations, f, indent=2)

    with open(os.path.join(args.output_dir, f'scaled_annotations.json'), 'w') as f:
                json.dump(annotations, f, indent=2)

