#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_distributed_pretrain_for_gpu.sh DEVICE_NUM EPOCH_SIZE DATA_DIR SCHEMA_DIR"
echo "for example: bash scripts/run_distributed_pretrain_for_gpu.sh 8 40 /path/zh-wiki/ [/path/Schema.json](optional)"
echo "It is better to use absolute path."
echo "=============================================================================================================="

RANK_SIZE=$1
EPOCH_SIZE=$2
DATA_DIR=$3
SCHEMA_DIR=$4

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
  python run_pretrain.py        \
    --device_target="GPU"      \
    --distribute="true"        \
    --epoch_size=$EPOCH_SIZE    \
    --enable_save_ckpt="true"    \
    --enable_lossscale="true"    \
    --do_shuffle="true"        \
    --enable_data_sink="true"    \
    --data_sink_steps=20        \
    --load_checkpoint_path=""      \
    --save_checkpoint_steps=10000  \
    --save_checkpoint_num=1      \
    --data_dir=$DATA_DIR      \
    --schema_dir=$SCHEMA_DIR > log.txt 2>&1 &

