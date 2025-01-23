# SSNet

## 1. Set up the environment
Build the environment by referring to ssnet_env.yml.  
CUDA, cuDNN should be versioned so that sensorflow 2.12.0 can be used.  
(CUDA 11.8, cuDNN 8.6)
  
~~~
conda env create -f ssnet_env_new.yml
conda activate ssnet_new
~~~



## 2. Prepare the dataset
Prepare your data.  
For training, the data consists of three columns: seq, da, and ie (ie is optional).  
For prediction, the data consists of only seq.  
The seq contains sequences in capital letters, da contains a list of 0/1/2 indicating whether each base is a donor/acceptor/none, and ie contains a list of 1/0 indicating whether each base is an exon/intron.  

We can split/padding this so that each line is 100k, and convert seq so that A/C/G/T is 1/2/3/4.  
For padding, use 0 for seq, 3 for da, and -1 for ie.  
Convert to tfrecord format and save the data file.  

| seq | da | ie |
| ---- | ---- | ---- |
| AGCTAGTGT...GTAC | 2222022212...22 | 11111000...0111 |


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
    --batch_size 16 \
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

## 4. Prediction

If you want to use trained models, you can download them from the below link.  
[SSNet_base](https://drive.google.com/file/d/1_y6PM3OKtx80WYLboI3cWuWIbhqc-ju-/view?usp=sharing)  
[SSNet_gtex](https://drive.google.com/file/d/1qnPg50LiWZ9hS1SKSitPeICUDGqTJxT9/view?usp=sharing)  
[SSNet_gtex_pangolin](https://drive.google.com/file/d/1wR9xkkZeTnxyQvhiRQagWbxvN16RkLsI/view?usp=sharing)  
[SSNet_pangolin](https://drive.google.com/file/d/1xlgH99UkFeH5W4osXjljCdqeC7uSCECj/view?usp=sharing)  
[SSNet_pangolin_gtex](https://drive.google.com/file/d/1eywvsURfKi5ONktMZlj41Db73Q_CJbWl/view?usp=sharing)  

### Prediction Only
~~~
python prediction.py \
    --test_file dataset/test.tfrecord \
    --model_path model/model_200_371200.h5 \
    --summary_dir sample_pred \
    --res_file pred.pkl \
    --gpu 0 \
    --batch_size 16 \
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
    --summary_dir sample_pred \
    --res_file attn.pkl \
    --gpu 0 \
    --batch_size 16 \
~~~

If you want to include attention in the prediction, give attention as a flag.  
sample_pred/attn.pkl consists of three columns: pred, global, and local.  
'global' contains global attention and ‘local’ contains local attention.
