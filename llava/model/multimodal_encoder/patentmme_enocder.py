import torch
import torch.nn as nn

from transformers import AutoProcessor, AutoConfig, LayoutLMv3ImageProcessor
from .modelling_layoutv3 import LayoutLMv3Model

class PatentMMETower(nn.Module):
    processor = LayoutLMv3ImageProcessor.from_pretrained("microsoft/layoutlmv3-large", apply_ocr=False)

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        
        self.vision_tower_ckp_pth = args.mm_vision_tower

        self.cfg_only = AutoConfig.from_pretrained("microsoft/layoutlmv3-large")

    def load_model(self):
        print('Initializing PatentMME encoder')
        self.processor.size = {'height': 384, 'width': 384}
        self.image_processor = self.processor


        self.cfg_only.input_size = 384
        self.cfg_only.patch_size = 16

        self.vision_tower = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-large", config=self.cfg_only, ignore_mismatched_sizes=True)

        ###
        sd = torch.load('checkpoints/patentmme/mlm_lamim_pc/last.ckpt', map_location='cpu')['state_dict']
        sd = {k.replace('model.', ''): v for k, v in sd.items()}
        self.vision_tower.load_state_dict(sd, strict=False)
        ###

        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images, ocr_input_ids, ocr_attention_mask, ocr_bboxes):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(pixel_values=images.to(device=self.device, dtype=self.dtype), input_ids=ocr_input_ids.to(device=self.device),
                                                    attention_mask=ocr_attention_mask.to(device=self.device, dtype=self.dtype), bbox=ocr_bboxes.to(device=self.device), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)


        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.cfg_only

    @property
    def hidden_size(self):
        return 1024

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
