# Finetune Code

## Classification

First, you need to organize the training data into the following format:

> root_data_path <br>
&emsp;&emsp;|---train.csv <br>
&emsp;&emsp;|---val.csv (Optional) <br>
&emsp;&emsp;|---test.csv (Optional) <br>

Each csv file needs to contain the following five columns：

| id_0   	| seq_0 |     id_1	 | seq_1   	| label |
| :--: 	| :--: | :--: | :--: | :--:	 |
| the first protein id | the first protein sequence | the second protein id | the second protein sequence | int label |


After that, you need to organize the data into Record format：
```
python generate_seq_for_classification_2x.py --data_dir <root_data_path> --vocab_file <vocab_file_path> --output_dir <save_mr_data_path> --max_seq_length <max_seq_length> --do_train <if_process_train.csv> --do_eval <if_process_val.csv> --do_test <if_process_test.csv>
```

Then, use the following scirpt to train and evaluate model:
```
python mpbert_classification.py --config_path <select_a_config_file> --do_train <if_train_the_model> --do_eval <if_evaluate_the_model> --description classification --num_class <class_num> --epoch_num <epoch_num> --data_url <mr_data_path> --load_checkpoint_url <pretrain_model> --output_url <save_model_dir> --task_name <save_model_name> 
```

## Site Predict

First, you need to organize the training data into the following format:

> root_data_path <br>
&emsp;&emsp;|---train.csv <br>
&emsp;&emsp;|---val.csv (Optional) <br>
&emsp;&emsp;|---test.csv (Optional) <br>

Each csv file needs to contain the following five columns：

| id_0   	| seq_0 |     label_0	 | id_1   	| seq_1 | label_1	 |
| :--: 	| :--: | :--: | :--: | :--:	 | :--:	 |
| the first protein id | the first protein sequence | the first label | the second protein id | the second protein sequence | the second label |

label_format:0001111000000..., consistent with the length of the sequence

After that, you need to organize the data into Record format：
```
python generate_seq_for_sequence_2x.py --data_dir <root_data_path> --vocab_file <vocab_file_path> --output_dir <save_mr_data_path> --max_seq_length <max_seq_length> --do_train <if_process_train.csv> --do_eval <if_process_val.csv> --do_test <if_process_test.csv>
```

Then, use the following scirpt to train and evaluate model:
```
python mpbert_sequence.py --config_path <select_a_config_file> --do_train <if_train_the_model> --do_eval <if_evaluate_the_model> --description classification --num_class <class_num> --epoch_num <epoch_num> --data_url <mr_data_path> --load_checkpoint_url <pretrain_model> --output_url <save_model_dir> --task_name <save_model_name> 
```
