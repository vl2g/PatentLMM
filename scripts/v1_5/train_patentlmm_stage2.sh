#!/bin/bash

deepspeed  --master_port=29502 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "liuhaotian/llava-v1.5-7b" \
    --version v1 \
    --data_path DATASET/llava_json/brief_train \
    --image_folder DATASET/images \
    --vision_tower patentmme \
    --pretrain_mm_mlp_adapter checkpoints/mlm_mim_pch_hupd_llama/pretrain_brief/checkpoint-1000/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir 'checkpoints/mlm_mim_pch_hupd_llama/brief' \
    --num_train_epochs 25 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 8 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to none 2>&1 | tee mlm_mim_pch_hupd_llama.log
