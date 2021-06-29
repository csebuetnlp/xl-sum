#!/bin/bash

source activate "${BASE_DIR}/env"
# for ozstar only; the model must
# be cached if this variable is set
export LINK_CACHE_ONLY=true 

# training settings
export max_steps=50000
export save_steps=2500
export logging_steps=100

# validation settings
export evaluation_strategy="no"

# model settings
export model_name="google/mt5-base"

# optimization settings
export learning_rate=1
export warmup_steps=5000
export gradient_accumulation_steps=16
export weight_decay=0.01
export lr_scheduler_type="transformer"
export label_smoothing_factor=0.1

# misc. settings
export seed=1234

# input / output settings
export input_dir="${BASE_DIR}/XLSum_input/multilingual"
export output_dir="${BASE_DIR}/XLSum_output/multilingual"

# batch / sequence sizes
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export MAX_SOURCE_LENGTH=512
export MAX_TARGET_LENGTH=84

# multilingual settings
export upsampling_factor=0.5

# optional arguments
optional_arguments=(
    "--logging_first_step"
    "--cache_dir ${BASE_DIR}/cache_dir"
)

export WANDB_PROJECT="XLSum-multilingual"
export WANDB_WATCH=false
export WANDB_MODE="dryrun"
export WANDB_DISABLED=true

python -m torch.distributed.launch \
		--nproc_per_node=$NPROC_PER_NODE \
		--nnodes=$SLURM_JOB_NUM_NODES \
		--node_rank=$SLURM_PROCID \
		--master_addr="$PARENT" --master_port="$MPORT" "${BASE_DIR}/pipeline.py" \
    --model_name_or_path $model_name \
    --data_dir $input_dir --output_dir $output_dir \
    --learning_rate=$learning_rate --warmup_steps $warmup_steps --gradient_accumulation_steps $gradient_accumulation_steps \
    --weight_decay $weight_decay --lr_scheduler_type $lr_scheduler_type --adafactor --label_smoothing_factor $label_smoothing_factor \
    --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE --logging_steps $logging_steps \
    --max_source_length $MAX_SOURCE_LENGTH --max_target_length $MAX_TARGET_LENGTH \
    --upsampling_factor $upsampling_factor --seed $seed --overwrite_output_dir \
    --max_steps $max_steps --save_steps $save_steps \
    --evaluation_strategy $evaluation_strategy --do_train \
    $(echo ${optional_arguments[@]})

