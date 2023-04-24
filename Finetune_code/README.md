# Finetune Code

## Classification

First, you need to organize the training data into the following format:

> root_data_path <br>
&emsp;&emsp;|---train.csv <br>
&emsp;&emsp;|---val.csv (Optional) <br>
&emsp;&emsp;|---test.csv (Optional) <br>

Each csv file needs to contain the following five columns：

| id_0   	| seq_0 |     id_1	 | seq_1   	| label |
| :-----: 	| :--: | :--: | :--: | :-------:	 |
| the first protein id | the first protein sequence | the second protein id | the second protein sequence | int label |


After that, you need to organize the data into Record format：
```
python generate_seq_for_classification_2x.py --data_dir <root_data_path> --vocab_file <vocab_file_dir> --output_dir <save_mr_data_dir> --max_seq_length <max_seq_length> --do_train <if_process_train.csv> --do_eval <if_process_val.csv> --do_test <if_process_test.csv>
```

