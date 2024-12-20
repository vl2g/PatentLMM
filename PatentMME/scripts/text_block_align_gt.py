import argparse
import json
import numpy as np
import os
import tqdm
import pandas as pd


def bbox_to_patches(args, bbox):
    n_patches_x = args.input_size / args.patch_size

    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[2]
    y_max = bbox[3]
    # +1 for indexing starting from 1 and not 0
    strt_patch_id = (y_min//args.patch_size)*n_patches_x + (x_min//args.patch_size) + 1
    end_patch_id = (y_max//args.patch_size)*n_patches_x + (x_max//args.patch_size) + 1
    
    i = -1
    word_patches = []
    while True:
        i += 1
        new_row = int(strt_patch_id + n_patches_x * i)
        if new_row >= end_patch_id:
            break
        # +1 for indexing starting from 1 and not 0
        # +1 to also include last value
        word_patches.extend(range(new_row, new_row + int((x_max - x_min)//args.patch_size) + 1))
    
    return word_patches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frcnn_scaled_anns", default='../annotations/combined_scaled_annotations.json')
    parser.add_argument("--ocr_file", default='../annotations/combined_ocr.json')
    parser.add_argument("--input_size", type=int, default=384, help='processed image dimension')
    parser.add_argument("--patch_size", type=int, default=16, help='patch dimension from ocr-VQGAN')
    parser.add_argument("--output_dir", default='../sample_data/', help='path to the directory where text-block alignment ground truth annotations are to be saved')

    args = parser.parse_args()

    with open(args.ocr_file, 'r') as f:
        ocr = json.load(f)
    
    with open(args.frcnn_scaled_anns, 'r') as f:
        block_annotations = json.load(f)

    tba_gt_anns = {}

    common_files = set(ocr.keys()).intersection(set(block_annotations.keys()))

    for im_id, im_name in enumerate(tqdm.tqdm(common_files)):
        im_ocr_bboxes = ocr[im_name]['bbox'][0]
        im_blck_anns = block_annotations[im_name]
        
        # separate the category ids and bboxes for image
        im_blck_bboxes = []
        im_blck_ids = []
        for ctg_anns in im_blck_anns:
            im_blck_ids.append(ctg_anns['category_id'])
            if 384.0 in ctg_anns['bbox']:
                ctg_anns['bbox'] = [x - 0.001 for x in ctg_anns['bbox']]
            im_blck_bboxes.append(ctg_anns['bbox'])

        print(im_blck_bboxes)
        
        annotations_dataframe = pd.DataFrame(im_blck_bboxes, columns=['x_min', 'y_min', 'x_max', 'y_max'])
        annotations_dataframe['category_id'] = im_blck_ids

        annotations_dataframe = annotations_dataframe.sort_values(by=['y_min', 'x_min', 'y_max', 'x_max'])

        im_blck_bboxes = annotations_dataframe[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
        im_blck_ids = annotations_dataframe['category_id'].values.tolist()

        print(im_name)
        print(im_blck_bboxes)
        
        # get the indices of node in category ids, and get the bboxes at those indices. This gives bboxes of all nodes in this image
        node_indices = np.where(np.array(im_blck_ids)==0)[0].tolist()
        
        patches_per_node = []
        for node in node_indices:
            # map the bbox of that node to the patches it spans across
            patches_per_node.append(bbox_to_patches(args, im_blck_bboxes[node]))
        
        word_wise_node_ids = []
        for word_id, word_bbox in enumerate(im_ocr_bboxes):
            # print("word bbox ========", (np.array(word_bbox)/1000)*args.input_size)
            word_patches = bbox_to_patches(args, ((np.array(word_bbox)/1000)*args.input_size).tolist())
            # print("word patches ========", word_patches)
            corresponding_node_id = -100
            corresponding_node_patches = []
            for node_id, node_patch in enumerate(patches_per_node):
                if len(set(node_patch) - set(word_patches)) != len(node_patch):
                    if len(corresponding_node_patches) != 0 and len(node_patch) < len(corresponding_node_patches):
                        corresponding_node_id = node_id
                        corresponding_node_patches = node_patch
                    elif len(corresponding_node_patches) == 0:
                        corresponding_node_id = node_id
                        corresponding_node_patches = node_patch
            
            word_wise_node_ids.append(corresponding_node_id)

        tba_gt_anns[im_name] = word_wise_node_ids

    with open(os.path.join(args.output_dir, f'tba_annotations_temp.json'), 'w') as f:
        json.dump(tba_gt_anns, f, indent=2)
