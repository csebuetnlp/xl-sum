We use a modified fork of [huggingface transformers](https://github.com/huggingface/transformers) for our experiments.

## Setup

```bash
$ git clone https://github.com/csebuetnlp/xl-sum
$ cd xl-sum/seq2seq
$ conda create python==3.7.9 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch -p ./env
$ conda activate ./env # or source activate ./env (for older versions of anaconda)
$ bash setup.sh 
```
* Use the newly created environment for running rest of the commands.

## Extracting data

Before running the extractor, place all the `.jsonl` files (`train`, `val`, `test`) for all the languages you want to work with, under a single directory (without any subdirectories). 

For example, to replicate our multilingual setup with all languages, run the following commands:

```bash
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fKxf9jAj0KptzlxUsI3jDbp4XLv_piiD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fKxf9jAj0KptzlxUsI3jDbp4XLv_piiD" -O XLSum_complete_v2.0.tar.bz2 && rm -rf /tmp/cookies.txt
$ tar -xjvf XLSum_complete_v2.0.tar.bz2
$ python extract_data.py -i XLSum_complete_v2.0/ -o XLSum_input/
```
This will create the source and target files for multilingual training within `XLSum_input/multilingual` and per language training and evaluation filepairs under `XLSum_input/individual/<language>`.


## Training & Evaluation

To see list of all available options, do `python pipeline.py -h`

### Multilingual training
* For multilingual training on single GPU, a minimal example is as follows:
```bash
$ python pipeline.py \
    --model_name_or_path "google/mt5-base" \
    --data_dir "XLSum_input/multilingual" \
    --output_dir "XLSum_output/multilingual" \
    --lr_scheduler_type="transformer" \
    --learning_rate=1 \
    --warmup_steps 5000 \
    --weight_decay 0.01 \ 
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=16  \
    --max_steps 50000 \
    --save_steps 5000 \
    --evaluation_strategy "no" \
    --logging_first_step \
    --adafactor \
    --label_smoothing_factor 0.1 \
    --upsampling_factor 0.5 \
    --do_train
```
* For multilingual training on multiple nodes / GPUs launch the script with `torch.distributed.launch`, i.e.
```bash
$ python -m torch.distributed.launch \
    --nproc_per_node=<NPROC_PER_NODE> \
    --nnodes=<NUM_NODES> \
    --node_rank=<PROCID> \
    --master_addr=<ADDR> \
    --master_port=<PORT> \
    pipeline.py ... 
```
To replicate our setup on 8 GPUs (4 nodes with 2 `NVIDIA TESLA P100` GPUs each) using SLURM, refer to [job.sh](job.sh) and [distributed_trainer.sh](distributed_trainer.sh) 

### Per language training
* Minimal training example (for example, on`Bengali`) on a single GPU is given below:
```bash
$ python pipeline.py \
    --model_name_or_path "google/mt5-base" \
    --data_dir "XLSum_input/individual/bengali" \
    --output_dir "XLSum_output/individual/bengali" \
    --lr_scheduler_type="linear" \
    --learning_rate=5e-4 \
    --warmup_steps 100 \
    --weight_decay 0.01 \ 
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=16  \
    --num_train_epochs=10 \
    --save_steps 100 \
    --predict_with_generate \
    --evaluation_strategy "epoch" \
    --logging_first_step \
    --adafactor \
    --label_smoothing_factor 0.1 \
    --do_train \
    --do_eval
```  
Hyperparameters such as `warmup_steps` should be updated according to the language. For a detailed example, refer to [trainer.sh](trainer.sh)

### Evaluation
* To calculate rouge scores on test sets (for example on `Hindi`) using a trained model, use the following snippet:

```bash
$ python pipeline.py \
    --model_name_or_path <path/to/trained/model/directory> \
    --data_dir "XLSum_input/individual/hindi" \
    --output_dir "XLSum_output/individual/hindi" \
    --rouge_lang "hindi" \ 
    --predict_with_generate \
    --length_penalty 0.6 \
    --no_repeat_ngram_size 2 \
    --max_source_length 512 \
    --test_max_target_length 84 \
    --do_predict
```
For a detailed example, refer to [evaluate.sh](evaluate.sh)
