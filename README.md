# SSNet

This repository contains the implementation of a splice site prediction model.   
It provides the code for reproducing the main model, along with example scripts for running predictions on genomic sequences. 

## 1. Set up the environment
Build the environment by referring to ssnet_env.yml.  
CUDA and cuDNN must be compatible with TensorFlow 2.12.0 (CUDA 11.8 and cuDNN 8.6).
~~~
conda env create -f ssnet_env.yml
conda activate ssnet
~~~


## 2. Prepare the dataset
Prepare your data.  
For training, the data consists of three columns: seq, da, and ie (ie is optional).  
For prediction, the data consists of only seq.  
The seq contains sequences in capital letters, da contains a list of 0/1/2 indicating whether each base is a donor/acceptor/none, and ie contains a list of 1/0 indicating whether each base is an exon/intron.  

| seq | da | ie |
| ---- | ---- | ---- |
| AGCTAGTGT...GTAC | 2222022212...22 | 11111000...0111 |

We can split/padding this so that each line is 100k, and convert seq so that A/C/G/T is 1/2/3/4.  
For padding, use 0 for seq, 3 for da, and -1 for ie.  
Convert to tfrecord format and save the data file.  

There are sample files (sample_train.tsv, sample_val.tsv, sample_test.tsv) in dataset.
You can generate tfrecord file from these sample files by running make_dataset.py.
~~~
python make_dataset.py \
    --data_dir dataset/ \
    --in_file sample_train.tsv \
    --out_file sample_train.tfrecord \
~~~

## 3. Training
### From the beginning
~~~
python training.py \
    --train_file dataset/train.tfrecord \
    --valid_file dataset/valid.tfrecord \
    --checkpoint_dir checkpoint/ \
    --summary_dir summary/ \
    --gpu 0 \
    --batch_size 16 \
    --summary_step 100 \
    --num_epoch 50 \
    --gamma 2.0 \
    --alpha1 0.05 0.25 0.7 \
    --alpha2 8.0 8.0 3.0 1.0 \
    --lr 0.000050 \
~~~

gamma, alpha1, alpha2 are hyperparameters to the loss function.  
lr is the learning rate.  

The following log is displayed on the standard output.  
~~~
Start of epoch 1
Epochs: 01, Steps: 000001, loss_train: 1.0618e-01, acc_train: 0.3501, recall_train: 0.3108, precision_train: 0.0842, 

-----Validation-----
Epochs: 01, loss_valid: 1.0620e-01, acc_valid: 0.3495, recall_valid: 0.3145, precision_valid: 0.0841, 

-----Save Model-----
Saved Model : model_01_1.h5

-----Save Optimizer-----
Saved Optimizer : opt_01_1.pkl
~~~

The trained model and optimizer are stored under checkpoint.  

### Continuing from the last time
~~~
python training.py \
    --train_file dataset/sample.tfrecord \
    --valid_file dataset/sample.tfrecord \
    --checkpoint_dir checkpoint/ \
    --summary_dir summary/ \
    --gpu 0 \
    --batch_size 4 \
    --pseudo_batch_size 16 \
    --summary_step 100 \
    --num_epoch 10 \
    --gamma 2.0 \
    --alpha1 0.05 0.25 0.7 \
    --alpha2 8.0 8.0 3.0 1.0 \
    --lr 0.000050 \
    --model_path checkpoint/model_10_10.h5 \
    --opt_path checkpoint/opt_10_10.pkl \
    --start_epoch 10 \
    --start_step 10
~~~

To load a saved model and optimizer and start training from where it left off.  
Specify start_epoch and start_step along with model_path and opt_path.
When training on a different dataset (e.g., using the gtex/pangolin dataset after the gencode dataset), you only need to specify the model_path.

## 4. Prediction

If you want to use trained models, you can download them from the below link.  
[SSNet_base](https://drive.google.com/file/d/1_y6PM3OKtx80WYLboI3cWuWIbhqc-ju-/view?usp=sharing)  
[SSNet_gtex](https://drive.google.com/file/d/1qnPg50LiWZ9hS1SKSitPeICUDGqTJxT9/view?usp=sharing)  
[SSNet_gtex_pangolin](https://drive.google.com/file/d/1wR9xkkZeTnxyQvhiRQagWbxvN16RkLsI/view?usp=sharing)  
[SSNet_pangolin](https://drive.google.com/file/d/1xlgH99UkFeH5W4osXjljCdqeC7uSCECj/view?usp=sharing)  
[SSNet_pangolin_gtex](https://drive.google.com/file/d/1eywvsURfKi5ONktMZlj41Db73Q_CJbWl/view?usp=sharing)  

### Prediction Only from TFRecord File
~~~
python prediction.py \
    --test_file dataset/test.tfrecord \
    --model_path model/model_200_371200.h5 \
    --summary_dir res \
    --res_file pred.pkl \
    --gpu 0 \
    --batch_size 4 \
~~~

In model_path, put the path of the trained model.  
Prediction results are output to sample_pred/pred.pkl.
This file consists of one column, pred.

### Prediction with Attention
~~~
python prediction.py \
    --attention \
    --test_file dataset/sample.tfrecord \
    --model_path model/model_200_371200.h5 \
    --summary_dir res \
    --res_file attn.pkl \
    --gpu 0 \
    --batch_size 4 \
~~~

If you want to include attention in the prediction, give attention as a flag.  
sample_pred/attn.pkl consists of three columns: pred, global, and local.  
'global' contains global attention and ‘local’ contains local attention.

### Aberrant Splicing Prediction from VCF File
~~~
python prediction.py \
    --vcf \
    --test_file dataset/sample_diff.vcf \
    --annotation grch37.txt \
    --fasta_dir hg19 \
    --model_path model/model_200_371200.h5 \
    --summary_dir res \
    --res_file pred.tsv \
    --mask
    --gpu 0 \
~~~

If you want to make aberrant splicing prediction from vcf file, give vcf as as flag.
In model_path, put the path of the trained model.  
Prediction results are output to sample_pred/pred.tsv.  
This file consists of one column, pred_diff.  
This indicates the maximum difference in predicted values between the ref sequence and the alt sequence.  
The mask flag restricts increases in predicted values to non-splice sites in the annotation file and decreases to splice sites in the annotation file only when calculating the difference in predicted values.  
Specify the annotation file as [grch37.txt](https://github.com/Illumina/SpliceAI/blob/master/spliceai/annotations/grch37.txt) or [grch38.txt](https://github.com/Illumina/SpliceAI/blob/master/spliceai/annotations/grch37.txt) from [SpliceAI github](https://github.com/Illumina/SpliceAI/tree/master/spliceai), or an original annotation file in a comparable format.  
Specify the directory containing the fasta files separated by chromosome (e.g., chr1.fa) for fasta_dir.  
Fasta files are downloadable from [hg19](https://hgdownload.cse.ucsc.edu/goldenpath/hg19/chromosomes/) and [hg38](https://hgdownload.cse.ucsc.edu/goldenpath/hg38/chromosomes/).

### Prediction for Reproducing Results in the Paper
~~~
python prediction_reproduce.py \
    --test_file dataset/brca.tfrecord \
    --data_type BRCA \
    --model_path model/model_200_371200.h5 \
    --summary_dir res \
    --res_file pred.pkl \
    --gpu 0 \
    --batch_size 4 \
~~~

Please specify corresponding "data_type" for each dataset.
All the data types are below:
 - SpliceVarDB
 - BRCA
 - sscvdb
 - iravdb
 - DMD
 - clinvar
 - IgM
 - FAS
 - attn_insilico
 - BRCA_attn

## Contact
For questions or inquiries about this repository or the splice prediction model, please contact:

- Name: Yuna Miyachi  
- Email: miyachi-yuna728@g.ecc.u-tokyo.ac.jp
- GitHub: [https://github.com/yuna728](https://github.com/yuna728)