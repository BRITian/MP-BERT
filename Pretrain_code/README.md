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

