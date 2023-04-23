#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_distributed_pretrain_ascend.sh DATA_DIR RANK_TABLE_FILE"
echo "for example: bash scripts/run_distributed_pretrain_ascend.sh /path/dataset /path/hccl.json"
echo "It is better to use absolute path."
echo "For hyper parameter, please note that you should customize the scripts:
          '{CUR_DIR}/scripts/ascend_distributed_launcher/hyper_parameter_config.ini' "
echo "=============================================================================================================="
CUR_DIR=`pwd`
ulimit -s 302400
python ${CUR_DIR}/scripts/ascend_distributed_launcher/get_distribute_pretrain_cmd.py \
    --run_script_dir=${CUR_DIR}/run_pretrain.py \
    --hyper_parameter_config_dir=${CUR_DIR}/scripts/ascend_distributed_launcher/hyper_parameter_config.ini \
    --data_dir=$1 \
    --hccl_config_dir=$2 \
    --hccl_time_out=600 \
    --hccn_config_file='/etc/hccn.conf' \
    --cmd_file=distributed_cmd.sh

bash distributed_cmd.sh
