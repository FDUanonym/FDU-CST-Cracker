#!/usr/bin zsh
export CUDA_VISIBLE_DEVICES=0
MODEL_FLAGS="--attention_resolutions 32,16,8
            --class_cond True
            --rescale_timesteps True
            --diffusion_steps 1000
            --dropout 0.1
            --image_size 64
            --learn_sigma True
            --noise_schedule cosine
            --num_channels 192
            --num_head_channels 64
            --num_res_blocks 3
            --resblock_updown True
            --use_new_attention_order True
            --use_fp16 True
            --use_scale_shift_norm True"

python classifier_sample.py --attention_resolutions 32,16,8 \
            --class_cond True \
            --rescale_timesteps True \
            --diffusion_steps 100 \
            --dropout 0.1 \
            --image_size 64 \
            --learn_sigma True \
            --noise_schedule cosine \
            --num_channels 192 \
            --num_head_channels 64 \
            --num_res_blocks 3 \
            --resblock_updown True \
            --use_new_attention_order True \
            --use_fp16 True \
            --use_scale_shift_norm True \
      --classifier_scale 10 \
      --classifier_path ckpt/64x64_classifier.pt \
      --classifier_depth 4 \
      --model_path ckpt/64x64_diffusion.pt \
      --save_dir outputs $SAMPLE_FLAGS \
      --class_index 1 \
      --use_ddim False

