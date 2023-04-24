#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_standalone_pretrain_ascend.sh DEVICE_ID EPOCH_SIZE DATA_DIR SCHEMA_DIR"
echo "for example: bash scripts/run_standalone_pretrain_ascend.sh 0 40 /path/zh-wiki/ [/path/Schema.json](optional)"
echo "=============================================================================================================="

DEVICE_ID=$1
EPOCH_SIZE=$2
DATA_DIR=$3
SCHEMA_DIR=$4
ulimit -s 102400

mkdir -p ms_log 
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
python ${PROJECT_DIR}/../run_pretrain.py  \
    --distribute="false" \
    --epoch_size=$EPOCH_SIZE \
    --device_id=$DEVICE_ID \
    --enable_save_ckpt="true" \
    --enable_lossscale="true" \
    --do_shuffle="true" \
    --enable_data_sink="true" \
    --data_sink_steps=1 \
    --accumulation_steps=1 \
    --load_checkpoint_path="" \
    --save_checkpoint_steps=10000 \
    --save_checkpoint_num=1 \
    --data_dir=$DATA_DIR \
    --schema_dir=$SCHEMA_DIR > pretraining_log.txt 2>&1 &
