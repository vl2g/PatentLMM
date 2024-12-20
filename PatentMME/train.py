from model import PatentMME
from dataset import HUPDdataset, custom_collate
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import argparse
import datetime
import wandb
import json
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--hupd_imgs_dir", required=True)
    parser.add_argument("--hupd_train_val_dir", help='path to the folder containing train.txt and val.txt', default='/storage/PatentQA/gen_data/DATASET/splits/')
    parser.add_argument("--ocr_file", help='path to json file with word-level ocr', default='/storage/PatentQA/gen_data/DATASET/LayoutLM_ocr/combined_ocr.json')
    parser.add_argument("--scaled_anns_file", help='path to json file with scaled frcnn annotations', default='/storage/PatentQA/frcnn/PatentParsing_frcnn/annotations/scaled_annotations.json')
    parser.add_argument("--pch_gt_file", help='path to file containing ground truth annotations for patch classification head', default='/storage/PatentQA/frcnn/PatentParsing_frcnn/annotations/pch_annotations.json')
    parser.add_argument("--tna_gt_file", help='path to file containing ground truth annotations for text node alignment', default='/storage/PatentQA/frcnn/PatentParsing_frcnn/annotations/new_combined_tna_annotations.json')
    parser.add_argument("--tna_nodeid_patch_map_file", help='path to file containing nodeid to patch mapping for tna', default='/storage/PatentQA/frcnn/PatentParsing_frcnn/annotations/new_combined_nodeid_patches_map.json')
    parser.add_argument("--vqgan_ckpt_path", default='./ocr-vqgan/checkpoints/ocr-vqgan-f16-c16384-d256/last.ckpt')

    parser.add_argument("--vqgan_config_path", default='ocr-vqgan/configs/copy.yaml')

    parser.add_argument("--train_head_only", type=bool, default=False)
    parser.add_argument("--calc_pch_loss", type=bool, default=False)
    parser.add_argument("--calc_tna_loss", type=bool, default=False)
    parser.add_argument("--n_nodes", default=25, help='Max. number of nodes assumed per image')

    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing (default: 3)")

    #### defaults from the paper & ocr-vqgan
    parser.add_argument("--image_size", default=384)
    parser.add_argument("--patch_size", default=16)
    parser.add_argument("--n_classes_frcnn", default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.98))
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=10_000)
    parser.add_argument("--resume_from_ckpt", type=str, default='')

    args = parser.parse_args()
    
    ## DATA
    with open(os.path.join(args.hupd_train_val_dir, 'train.txt'), 'r') as f:
        ims_paths_train = f.read().splitlines()
    with open(os.path.join(args.hupd_train_val_dir, 'val.txt'), 'r') as f:
        ims_paths_val = f.read().splitlines()

    ims_paths_train = [img_pth for img_pth in ims_paths_train if (img_pth.split('/')[-1] in block_annotations and img_pth.split('/')[-1] in ocr_data)]
    ims_paths_val = [img_pth for img_pth in ims_paths_val if (img_pth.split('/')[-1] in block_annotations and img_pth.split('/')[-1] in ocr_data)]
    
    with open(args.ocr_file, 'r') as f:
        ocr_data = json.load(f)
    
    with open(args.scaled_anns_file, 'r') as f:
        block_annotations = json.load(f)

    
    with open(args.pch_gt_file, 'r') as f:
        pch_annotations = json.load(f)

    with open(args.tna_gt_file, 'r') as f:
        tna_annotations = json.load(f)
    
    with open(args.tna_nodeid_patch_map_file, 'r') as f:
        tna_nodeid_patch_map = json.load(f)

    train_ds = HUPDdataset(ims_paths_train, ocr_data,
                           block_annotations, pch_annotations,
                           tna_annotations, tna_nodeid_patch_map, 
                           args.image_size, args.patch_size, 
                           args.n_classes_frcnn, args.n_nodes, split='train')

    val_ds = HUPDdataset(ims_paths_val, ocr_data,
                         block_annotations, pch_annotations,
                         tna_annotations, tna_nodeid_patch_map,
                         args.image_size, args.patch_size, 
                         args.n_classes_frcnn, args.n_nodes, split='val')
    

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate)
    
    model = PatentMME(args.vqgan_config_path, args.vqgan_ckpt_path,
                           args.calc_pch_loss, args.calc_tna_loss,
                           args.train_head_only, args.image_size, 
                           args.patch_size, args.n_classes_frcnn, 
                           args.n_nodes, args.lr, args.betas, 
                           args.warmup_steps)
    
    if args.resume_from_ckpt != '':
        ckpt_path = args.resume_from_ckpt
        sd = torch.load(ckpt_path, map_location='cpu')['state_dict']
        model.load_state_dict(sd)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=f'checkpoints_pch={args.calc_pch_loss}_tna={args.calc_tna_loss}',
            monitor='val_total_loss',
            filename='PatentMME_new-epoch{epoch:02d}-val_loss={val_total_loss:.2f}',
            save_top_k=3,
            save_last=True
        )
    ]

    logger = WandbLogger(name=f'PatentMME_lr={args.lr}_bs={args.batch_size}_me={args.max_epochs}_pch={args.calc_pch_loss}_tna={args.calc_tna_loss}_train_head={args.train_head_only}', project="PatentMME")

    trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epochs,
                         check_val_every_n_epoch=1, callbacks=callbacks,
                         strategy=pl.strategies.DDPStrategy(timeout=datetime.timedelta(seconds=3600), find_unused_parameters=True),
                         logger=logger,
                         accumulate_grad_batches=args.grad_accumulation_steps)
    if trainer.global_rank == 0:
        logger.experiment.config.update(args)

    trainer.fit(model, train_dl, val_dl)
