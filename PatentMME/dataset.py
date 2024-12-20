import argparse
import os
from PIL import Image
import json
import torch
from torch.utils.data import Dataset, dataloader
from torch.nn.utils.rnn import pad_sequence


def custom_collate(batch):
    images = [b.pop('image') for b in batch]
    ocr_input_ids = [b.pop('ocr_input_ids') for b in batch]
    ocr_attention_mask = [b.pop('ocr_attention_mask') for b in batch]
    ocr_bboxes = [b.pop('ocr_bboxes') for b in batch]
    # block_ids = [b.pop('block_ids') for b in batch]
    # block_bboxes = [b.pop('block_bboxes') for b in batch]
    pch_labels = torch.stack([b.pop('pch_labels') for b in batch], dim=0)
    
    tna_labels = [b.pop('tna_labels') for b in batch]

    tna_nodeid_patch_map = torch.stack([b.pop('tna_nodeid_patch_map') for b in batch])
    
    ocr_input_ids = pad_sequence(ocr_input_ids, batch_first=True, padding_value=1)
    ocr_attention_mask = pad_sequence(ocr_attention_mask, batch_first=True, padding_value=0)
    ocr_bboxes = pad_sequence(ocr_bboxes, batch_first=True, padding_value=0)
    tna_labels = pad_sequence(tna_labels, batch_first=True, padding_value=-100)

    max_tna_node = min(torch.max(tna_labels) + 1, 25)
    
    tna_labels[tna_labels > max_tna_node - 1] = -100

    tna_nodeid_patch_map = tna_nodeid_patch_map[:, :max_tna_node, :]

    # image_names = [b.pop('im_name') for b in batch]
    
    collated_batch = dataloader.default_collate(batch)

    collated_batch['images'] = images
    collated_batch['ocr_input_ids'] = ocr_input_ids
    collated_batch['ocr_attention_mask'] = ocr_attention_mask
    collated_batch['ocr_bboxes'] = ocr_bboxes
    # collated_batch['block_ids'] = block_ids
    # collated_batch['block_bboxes'] = block_bboxes
    collated_batch['pch_labels'] = pch_labels
    collated_batch['tna_labels'] = tna_labels
    collated_batch['tna_nodeid_patch_map'] = tna_nodeid_patch_map
    # collated_batch['image_names'] = image_names

    return collated_batch





class HUPDdataset(Dataset):
    def __init__(self, ims_paths, ocr_data, block_annotations, pch_annotations, tna_annotations,
                 tna_nodeid_patch_map, image_size, patch_size, n_classes_frcnn, n_nodes, split='train'):
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_classes_frcnn = n_classes_frcnn
        self.n_nodes = n_nodes

        self.ims_paths = ims_paths
        # if ds_size != -1:
        #     self.ims_paths = self.ims_paths[:ds_size]
        self.ocr = ocr_data
        self.block_annotations = block_annotations
        self.pch_annotations = pch_annotations
        self.tna_annotations = tna_annotations
        self.tna_nodeid_patch_map = tna_nodeid_patch_map

        
    def __len__(self):
        return len(self.ims_paths)

    def __getitem__(self, idx):
        try:
            im_pth = self.ims_paths[idx]
            im_name = im_pth.split('/')[-1]
            # im_pth = self.ims_paths[idx] + '.png'
            # im_name = im_pth

            image = Image.open(im_pth)
            # image = Image.open(os.path.join('/storage/PatentQA/gen_data/DATASET/images_combined/', f'{im_name}')).convert('RGB')

            im_pch_ann = self.pch_annotations[im_name]

            im_tna_ann = self.tna_annotations[im_name]
            im_tna_nodeid_patch_map = self.tna_nodeid_patch_map[im_name]
            
            pch_labels = torch.zeros((self.image_size//self.patch_size)**2, self.n_classes_frcnn)
            for cls, ls_patches in im_pch_ann.items():
                pch_labels[list(map(lambda x: x - 1, ls_patches)), int(cls)] = 1

            ocr_input_ids = self.ocr[im_name]['input_ids'][0]
            ocr_attention_mask = self.ocr[im_name]['attention_mask'][0]
            ocr_bboxes = self.ocr[im_name]['bbox'][0]

            # +1 for nodes beyond n_nodes
            # tna_labels = [-100]*len(im_tna_ann)
            # for word_id, word_node in enumerate(im_tna_ann):
            #     if word_node == -100:
            #         tna_labels[word_id] = self.n_nodes
            #     elif word_node >= self.n_nodes:
            #         tna_labels[word_id] = self.n_nodes
            #     else:
            #         tna_labels[word_id] = word_node
            # tna_labels[0] = -100        # CLS token
            # tna_labels[-1] = -100       # SEP token

            tna_labels = im_tna_ann
            
            if len(ocr_input_ids) > 512:
                ocr_input_ids = ocr_input_ids[:511] + [ocr_input_ids[-1]]
                ocr_attention_mask = ocr_attention_mask[:511] + [ocr_attention_mask[-1]]
                ocr_bboxes = ocr_bboxes[:511] + [ocr_bboxes[-1]]
                tna_labels = im_tna_ann[:511] + [im_tna_ann[-1]]
            
            tna_nodeid_patch_map_tensor = torch.zeros(self.n_nodes, (self.image_size//self.patch_size)**2)
            for node, patches in im_tna_nodeid_patch_map.items():
                if int(node) < self.n_nodes:
                    tna_nodeid_patch_map_tensor[int(node), [p-1 for p in patches]] = 1
            
            # if torch.count_nonzero(tna_nodeid_patch_map_tensor.sum(-1, keepdim=True)) != self.n_nodes:
            #     print('image_name:', im_name)
            #     print('node_patch_mapping:', tna_nodeid_patch_map_tensor)
                
            sample = {
                'image': image,
                'ocr_input_ids': torch.tensor(ocr_input_ids),
                'ocr_attention_mask': torch.tensor(ocr_attention_mask),
                'ocr_bboxes': torch.tensor(ocr_bboxes),
                'pch_labels': pch_labels,
                'tna_labels': torch.tensor(tna_labels),
                'tna_nodeid_patch_map': tna_nodeid_patch_map_tensor,
                'im_name': im_name
            }

            return sample
        except Exception as e:
            print(e)
            return self.__getitem__(idx-1)
