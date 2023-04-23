# Pretrain Code


This section contains the data processing code for pre-training and the pre-training code <br>


## Pre-train Data Processing

First you need to build a serial relationship dataset using a tool such as ProtENN
Then organize your input data in the following format

> root_data_path <br>
&emsp;&emsp;| <br>
&emsp;&emsp;|---folder1 <br>
&emsp;&emsp;|&emsp;&emsp;|---file_name_1.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---file_name_2.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---... <br>

where each file_name_n.fasta represents the same type of protein

Usage:
```
python generate_pretrain.py --input_file <root_data_path> --output_file <output_path> --vocab_file <vocab_file_path> --max_seq_length <max_seq_length> --max_predictions_per_seq <max_predictions_per_seq>
```

**Attention**
* The processing of the pre-training data depends on your device, typically it takes around 20 days to process the data (with pfam predictions taking around 1-2 weeks, GPU accelerated predictions are recommended, and data processing to training data takes around 1-2 weeks)
* Keep an eye on your device memory, you may need to limit the total amount of data you can process in a single session

## Pre-training
#### Run on Ascend

```
# run standalone pre-training example
bash scripts/run_standalone_pretrain_ascend.sh 0 1 <mindrecord_path>

# run distributed pre-training example
python scripts/ascend_distributed_launcher/get_distribute_pretrain_cmd.py --run_script_dir ./scripts/run_distributed_pretrain_ascend.sh --hyper_parameter_config_dir ./scripts/ascend_distributed_launcher/hyper_parameter_config.ini --data_dir <mindrecord_path> --hccl_config /path/hccl.json --cmd_file ./distributed_cmd.sh
bash scripts/run_distributed_pretrain_ascend.sh <mindrecord_path> /path/hccl.json
```

You should create hccl.conf according to the [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools) help documentation

#### running on GPU

```
# run standalone pre-training example
bash scripts/run_standalone_pretrain_for_gpu.sh 0 1 <mindrecord_path>

# run distributed pre-training example
bash scripts/run_distributed_pretrain_for_gpu.sh <mindrecord_path>
```

The pre-training module of MP-BERT is based on the MindSpore BERT modification, but the script to start run the pre-training is the same as the MindSpore BERT, for more information on this please visit: 
https://gitee.com/mindspore/models/blob/master/official/nlp/Bert/README.md



