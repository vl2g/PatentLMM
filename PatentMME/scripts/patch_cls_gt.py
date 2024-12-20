import argparse
import json
import tqdm
import os
from PIL import Image
import numpy as np

def coordinate_patch_map(args, bboxes, category_ids):
    n_patches_x = args.input_size / args.patch_size
    img_cls_dist = {}

    # iterate through bounding boxes for different categories
    for nbbox, ctg_bbox in enumerate(bboxes):
        x_min = ctg_bbox[0]
        y_min = ctg_bbox[1]
        x_max = min(ctg_bbox[2], 383)
        y_max = min(ctg_bbox[3], 383)
        # +1 for indexing starting from 1 and not 0
        strt_patch_id = (y_min//args.patch_size)*n_patches_x + (x_min//args.patch_size) + 1
        end_patch_id = (y_max//args.patch_size)*n_patches_x + (x_max//args.patch_size) + 1
        
        i = -1
        category_patches = []
        while True:
            i += 1
            new_row = int(strt_patch_id + n_patches_x * i)
            if new_row > end_patch_id:
                break
            # +1 for indexing starting from 1 and not 0
            # +1 to also include last value
            category_patches.extend(range(new_row, new_row + int(x_max - x_min)//args.patch_size + 1))
        
        if category_ids[nbbox] not in img_cls_dist:
            img_cls_dist[category_ids[nbbox]] = []
        img_cls_dist[category_ids[nbbox]].extend(category_patches)

    for cls, patch_ids in img_cls_dist.items():
        img_cls_dist[cls] = list(set(patch_ids))

    return img_cls_dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, default=384, help='processed image dimension')
    parser.add_argument("--patch_size", type=int, default=16, help='patch dimension from ocr-VQGAN')
    parser.add_argument("--frcnn_scaled_anns_file", default='annotations/scaled_annotations.json', help='path to the scaled frcnn annotations file')
    parser.add_argument("--output_dir", default='annotations', help='path to the directory where patch classification ground truth annotations are to be saved')

    args = parser.parse_args()

    # im_path = '/DATA/nakul/PatentQA/main_methodology/LayoutLMv3/images_new_patent_data/'

    with open(args.frcnn_scaled_anns_file, 'r') as f:
        annotations = json.load(f)
    
    pch_annotations = {}
    for i, img_id in enumerate(tqdm.tqdm(annotations)):
        # img = Image.open(im_path + img_id)
        # orig_width, orig_height = img.size
        ls_anns = annotations[img_id]
        bboxes = []
        category_ids = []
        for anns in ls_anns:
            # anns['bbox'][0][0] = (anns['bbox'][0][0]/orig_width)*args.input_size
            # anns['bbox'][0][1] = (anns['bbox'][0][1]/orig_height)*args.input_size
            # anns['bbox'][0][2] = (anns['bbox'][0][2]/orig_width)*args.input_size
            # anns['bbox'][0][3] = (anns['bbox'][0][3]/orig_height)*args.input_size
            bboxes.append(anns['bbox'])
            category_ids.append(anns['category_id'])
        
        pch_annotations[img_id] = coordinate_patch_map(args, bboxes, category_ids)
    
        if i % 500 == 0:
            with open(os.path.join(args.output_dir, f'pch_annotations.json'), 'w') as f:
                json.dump(pch_annotations, f, indent=2)

    with open(os.path.join(args.output_dir, f'pch_annotations.json'), 'w') as f:
        json.dump(pch_annotations, f, indent=2)
