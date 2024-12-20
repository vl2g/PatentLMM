python train.py --hupd_train_val_dir <../ocr-vqgan/ --> this should contain train.txt and val.txt>\
    --ocr_file <Combined ocr file, produced in ../PatentParsing_frcnn/annotations> \
    --scaled_anns_file <Combined scaled_annotations file, produced in ../PatentParsing_frcnn/annotations> \
    --pch_gt_file <Combined pch_gt file, produced in ../PatentParsing_frcnn/annotations> \
    --vqgan_ckpt_path <Path to last.ckpt of ocr-vqgan, must be in ..//ocr-vqgan/checkpoints> \
    --train_head_only True\
    --max_epochs 200 \
    --batch_size 1 \
    --num_workers 4
