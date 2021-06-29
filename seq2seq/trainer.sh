#!/bin/bash
# to use this script for multilingual setup, use the variables 
# from `distributed_trainer.sh`

# training settings
export num_train_epochs=10
export max_steps=0 # overrides epochs (must be 0 if using epochs)
export save_steps=500
export logging_steps=100

# validation settings
export evaluation_strategy="epoch" 

# model settings
export model_name="google/mt5-base"

# optimization settings
export learning_rate=5e-4 
export warmup_steps=250 # we used 10% of the total number of steps as warmup for monolingual training.
export gradient_accumulation_steps=16
export weight_decay=0.01
export lr_scheduler_type="linear"
export label_smoothing_factor=0.1

# misc. settings
export seed=1234

# input / output settings
export input_dir="XLSum_input/individual/bengali"
export output_dir="XLSum_output/individual/bengali"

# batch / sequence sizes
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export PER_DEVICE_EVAL_BATCH_SIZE=1
export MAX_SOURCE_LENGTH=512
export MAX_TARGET_LENGTH=84
export VAL_MAX_TARGET_LENGTH=$MAX_TARGET_LENGTH

# optional arguments
optional_arguments=(
    "--logging_first_step"
    "--cache_dir cache_dir/"
)

# optional for logging
# export WANDB_PROJECT="MT5-Experiments"
# export WANDB_WATCH=false
# export WANDB_MODE="dryrun"
export WANDB_DISABLED=true

python ./pipeline.py \
    --model_name_or_path $model_name \
    --data_dir $input_dir --output_dir $output_dir \
    --learning_rate=$learning_rate --warmup_steps $warmup_steps --gradient_accumulation_steps $gradient_accumulation_steps \
    --weight_decay $weight_decay --lr_scheduler_type $lr_scheduler_type --adafactor --label_smoothing_factor $label_smoothing_factor \
    --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE --per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    --max_source_length $MAX_SOURCE_LENGTH --max_target_length $MAX_TARGET_LENGTH --logging_steps $logging_steps \
    --val_max_target_length $VAL_MAX_TARGET_LENGTH --seed $seed --overwrite_output_dir \
    --num_train_epochs=$num_train_epochs --max_steps $max_steps --save_steps $save_steps \
    --evaluation_strategy $evaluation_strategy --predict_with_generate --do_train --do_eval \
    $(echo ${optional_arguments[@]})

