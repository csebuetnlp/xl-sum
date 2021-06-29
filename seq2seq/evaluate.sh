#!/bin/bash

# misc. settings
export seed=1234

# model settings
export model_name=<path/to/trained/model/directory>

# input / output settings
export input_dir="XLSum_input/individual/bengali"
export output_dir="XLSum_output/individual/bengali"

# batch / sequence sizes
export PER_DEVICE_EVAL_BATCH_SIZE=8
export MAX_SOURCE_LENGTH=512
export TEST_MAX_TARGET_LENGTH=84

# evaluation settings
export rouge_lang="bengali"
export eval_beams=4
export length_penalty=0.6
export no_repeat_ngram_size=2

# optional_arguments
optional_arguments=(
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
    --per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    --max_source_length $MAX_SOURCE_LENGTH --test_max_target_length $TEST_MAX_TARGET_LENGTH \
    --rouge_lang $rouge_lang --length_penalty $length_penalty --no_repeat_ngram_size $no_repeat_ngram_size \
    --eval_beams $eval_beams --seed $seed --overwrite_output_dir --predict_with_generate --do_predict \
    $(echo ${optional_arguments[@]})
