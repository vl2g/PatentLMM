import pytorch_lightning as pl
from transformers import AutoProcessor, AutoConfig, LayoutLMv3ImageProcessor
from patentmme_layoutlmv3 import LayoutLMv3Model
import torch.optim as optim
import torch
from transformers import get_linear_schedule_with_warmup

class PatentMME(pl.LightningModule):
    processor = LayoutLMv3ImageProcessor.from_pretrained("microsoft/layoutlmv3-large", apply_ocr=False)
    
    def __init__(self, vqgan_config_path, vqgan_ckpt_path, calc_pch_loss=False, calc_tba_loss = False, train_head_only=False, image_size=384, patch_size=16, n_classes_frcnn=5, n_nodes=25, lr=1e-4, betas=(0.9, 0.98), warmup_steps=10000):
        super(PatentMME, self).__init__()
        self.lr = lr
        self.betas = betas
        self.warmup_steps = warmup_steps

        self.processor.size = {'height': image_size, 'width': image_size}

        self.config = AutoConfig.from_pretrained("microsoft/layoutlmv3-large")
        self.config.input_size = image_size
        self.config.patch_size = patch_size
        self.config.n_patches_classes = n_classes_frcnn
        self.config.n_nodes = n_nodes
        self.config.txt_mask_prob = 0.3
        self.config.vis_mask_prob = 0.4
        self.config.n_patch_classes = 5
        self.config.vqgan_config_path = vqgan_config_path
        self.config.vqgan_ckpt_path = vqgan_ckpt_path
    
        self.model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-large", calc_pch_loss = calc_pch_loss, calc_tna_loss = calc_tba_loss, train_head_only=train_head_only, config=self.config, ignore_mismatched_sizes=True)
        
        self.save_hyperparameters()

    def training_step(self, batch, idx):
        images = batch['images']
        ocr_input_ids = batch['ocr_input_ids']
        
        ocr_attention_mask = batch['ocr_attention_mask']
        ocr_boxes = batch['ocr_bboxes']
        # block_ids = batch['block_ids']
        # block_bboxes = batch['block_bboxes']
        pch_labels = batch['pch_labels']
        # tba_labels = batch['tba_labels']

        pixel_values = self.processor(images, return_tensors='pt')['pixel_values']
        tna_labels = batch['tna_labels']
        tna_nodeid_patch_map = batch['tna_nodeid_patch_map']
        
        outputs = self.model(pixel_values=pixel_values.to(self.device), input_ids=ocr_input_ids.to(self.device),
                             attention_mask=ocr_attention_mask.to(self.device), bbox=ocr_boxes.to(self.device),
                             pch_gt_anns=pch_labels.to(self.device),
                             tna_gt_anns=tna_labels.to(self.device), nodeid_patch_map=tna_nodeid_patch_map.to(self.device))
                             
        loss_dict = outputs.loss
        log_dict = {f'train_{k}':v for k,v in loss_dict.items() if v is not None}
        self.log_dict(log_dict ,batch_size=ocr_input_ids.shape[0], prog_bar=True)


        return loss_dict['total_loss']

    def validation_step(self, batch, idx):
        images = batch['images']
        ocr_input_ids = batch['ocr_input_ids']
        
        ocr_attention_mask = batch['ocr_attention_mask']
        ocr_boxes = batch['ocr_bboxes']
        # block_ids = batch['block_ids']
        # block_bboxes = batch['block_bboxes']
        \
        
        pch_labels = batch['pch_labels']
        # tba_labels = batch['tba_labels']

        pixel_values = self.processor(images, return_tensors='pt')['pixel_values']
        tna_labels = batch['tna_labels']
        tna_nodeid_patch_map = batch['tna_nodeid_patch_map']
        
        outputs = self.model(pixel_values=pixel_values.to(self.device), input_ids=ocr_input_ids.to(self.device),
                             attention_mask=ocr_attention_mask.to(self.device), bbox=ocr_boxes.to(self.device),
                             pch_gt_anns=pch_labels.to(self.device),
                             tna_gt_anns=tna_labels.to(self.device), nodeid_patch_map=tna_nodeid_patch_map.to(self.device))
                             
        loss_dict = outputs.loss

        log_dict = {f'val_{k}':v for k,v in loss_dict.items() if v is not None}

        self.log_dict(log_dict, batch_size=ocr_input_ids.shape[0], prog_bar=True)

    
    def configure_optimizers(self):
        opt = optim.Adam(
            self.parameters(), lr=self.lr, betas=self.betas,
            weight_decay=1e-2
        )

        sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=self.warmup_steps, num_training_steps=1e12)

        return [opt], [sch]

