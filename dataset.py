from torch.utils.data import Dataset, dataloader
import torchvision.transforms as transforms
import os
from PIL import Image
import json
import torch
from torch.nn.utils.rnn import pad_sequence

def custom_collate(batch):
    desc = [b.pop('description') for b in batch]
    # ocr_text = [b.pop('ocr_text') for b in batch]
    ocr_input_ids = [b.pop('ocr_input_ids') for b in batch]
    ocr_attention_mask = [b.pop('ocr_attention_mask') for b in batch]
    ocr_bboxes = [b.pop('ocr_bboxes') for b in batch]
    fig_id = [b.pop('fig_id') for b in batch]
    img = [b.pop('img') for b in batch]

    ocr_input_ids = pad_sequence(ocr_input_ids, batch_first=True, padding_value=1)
    ocr_attention_mask = pad_sequence(ocr_attention_mask, batch_first=True, padding_value=0)
    ocr_bboxes = pad_sequence(ocr_bboxes, batch_first=True, padding_value=0)

    collated_batch = dataloader.default_collate(batch)

    collated_batch['description'] = desc
    # collated_batch['ocr_text'] = ocr_text
    collated_batch['ocr_input_ids'] = ocr_input_ids
    collated_batch['ocr_attention_mask'] = ocr_attention_mask
    collated_batch['ocr_bboxes'] = ocr_bboxes
    collated_batch['fig_id'] = fig_id
    collated_batch['img'] = img

    return collated_batch




def get_img_transforms(mode='default', im_size=[224, 224]):
    if mode == 'default':
        t = transforms.Compose([
            transforms.Resize(im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return t



class PatentDescDataset(Dataset):

    def __init__(self, split='train',
                 desc_type='detailed',
                 ocr_only=False,
                 ocr_file='../gen_data/DATASET/LayoutLM_ocr/combined_ocr.json',
                 data_dir='../gen_data/DATASET',
                 data_len=-1):

        assert split in ['train', 'val', 'test'], RuntimeError(f"Invalid dataset split {split}")
        self.split = split

        self.desc_type = desc_type

        split_files_loc = os.path.join(data_dir, 'splits')
        with open(os.path.join(split_files_loc, f'{split}.txt')) as f:
            self.fnames = set([x.strip() for x in f.readlines()])
        
        # truncated descriptions
        with open(os.path.join(data_dir, 'truncated_descriptions.json')) as f:
            self.descriptions = {k: v[desc_type].strip() for k,v in json.load(f).items() if v[desc_type].strip()!=''}
        
        self.fnames = list(self.fnames.intersection(set(self.descriptions.keys())))
        
        self.img_dir = os.path.join(data_dir, 'images_combined')

        with open(ocr_file, 'r') as f:
            self.ocr = json.load(f)
        # self.ocr_dir = os.path.join(data_dir, 'ocr_combined')
        # self.desc_dir = os.path.join(data_dir, 'desc_ocr_combined')

        self.num_samples = len(self.fnames) if data_len == -1 else data_len

        self.im_transform = get_img_transforms()

        self.ocr_only = ocr_only

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        fig_id = self.fnames[idx]

        img = None
        if not self.ocr_only:
            img = Image.open(os.path.join(self.img_dir, f'{fig_id}.png')).convert('RGB')
            # img = self.im_transform(img)

        # with open(os.path.join(self.ocr_dir, f'{fig_id}.json')) as f:
        #     ocr_text = '. '.join(json.load(f)['text'])
            
        ocr_input_ids = self.ocr[fig_id+'.png']['input_ids'][0]
        ocr_attention_mask = self.ocr[fig_id+'.png']['attention_mask'][0]
        ocr_bboxes = self.ocr[fig_id+'.png']['bbox'][0]

        if len(ocr_input_ids) > 512:
            ocr_input_ids = ocr_input_ids[:511] + [ocr_input_ids[-1]]
            ocr_attention_mask = ocr_attention_mask[:511] + [ocr_attention_mask[-1]]
            ocr_bboxes = ocr_bboxes[:511] + [ocr_bboxes[-1]]
        
        # with open(os.path.join(self.desc_dir, f'{fig_id}.json')) as f:
        #     description = json.load(f)
        
        res = {
            'fig_id': fig_id,
            'ocr_input_ids': torch.tensor(ocr_input_ids),
            'ocr_attention_mask': torch.tensor(ocr_attention_mask),
            'ocr_bboxes': torch.tensor(ocr_bboxes),
            'description': self.descriptions[fig_id],
	    'image_id': fig_id
        }

        if not self.ocr_only:
            res.update({'img': img})
        
        return res
