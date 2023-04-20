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
&emsp;&emsp;|---folder2 <br>
&emsp;&emsp;|&emsp;&emsp;|---file_name_1.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---file_name_2.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---... <br>
&emsp;&emsp;|---... <br>
&emsp;&emsp;|---folderN <br>
&emsp;&emsp;&emsp;&emsp;|---file_name_1.fasta <br>
&emsp;&emsp;&emsp;&emsp;|---file_name_2.fasta <br>
&emsp;&emsp;&emsp;&emsp;|---... <br>
 <br>
For example, when we use Pfamily, the composition is as follows

> root_data_path <br>
&emsp;&emsp;| <br>
&emsp;&emsp;|---folder1 <br>
&emsp;&emsp;|&emsp;&emsp;|---PF00001.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---PF00002.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---... <br>
&emsp;&emsp;|---folder2 <br>
&emsp;&emsp;|&emsp;&emsp;|---PF00001.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---PF00002.fasta <br>
&emsp;&emsp;|&emsp;&emsp;|---... <br>
&emsp;&emsp;|---... <br>
&emsp;&emsp;|---folderN <br>
&emsp;&emsp;&emsp;&emsp;|---PF00001.fasta <br>
&emsp;&emsp;&emsp;&emsp;|---PF00002.fasta <br>
&emsp;&emsp;&emsp;&emsp;|---... <br>

其中：
* root_data_path 是--input_file输入参数的值
* folder1 ... folderN的作用是防止一个Mindrecord文件的序列条数过多，导致数据处理过程中超出设备内存，当内存够大时可以只有一个folder1，folder的名称可以自定，也可以根据自己的设备自定一次读入多少个folder
* file_name_1.fasta ... file_name_N.fasta 是相同聚类的集合，例如PF00001.fasta是所有00001家族的蛋白，不同folder中的相同聚类虽然在不同的fasta文件，但需要保持名字一致，例如folder1和folder2都有PF00001.fasta文件
