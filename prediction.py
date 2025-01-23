import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision
import os
import argparse
import pickle
import pandas as pd

from model import SSNet

INPUT_LEN = 100000

def parse_record(example):    
    context_features = {
        "seq" : tf.io.FixedLenFeature([INPUT_LEN], dtype=tf.int64)
    }

    context_parsed = tf.io.parse_example(serialized=example, features=context_features)
    
    seq_raw = context_parsed["seq"]
    seq_onehot = tf.one_hot(seq_raw, 5)
    mask_n = tf.reduce_all(tf.equal(seq_onehot, 0.), axis=-1, keepdims=True)
    n_token = tf.tile(0.25 * tf.cast(mask_n, tf.float32),[1,4])
    # PAD:[0, 0, 0, 0], A:[1, 0, 0, 0], C:[0, 1, 0, 0,], G:[0, 0, 1, 0], T:[0, 0, 0, 1], N:[0.25, 0.25, 0.25, 0.25]
    seq = seq_onehot[:,1:] + n_token # 100000 * 4

    return {"seq": seq}

def build_model():
  conv_dim = 64
  conv_kernel = [4, 16, 64]
  num_c_layers = 2
  num_t_layers = 4
  d_comp = 64
  d_model = 512
  num_areas = 625
  num_heads_comp = 2
  num_heads = 8
  dff_comp = 128
  dff = 512
  pe_input = INPUT_LEN

  model = SSNet(conv_dim, conv_kernel, num_c_layers, num_t_layers, d_comp, d_model, num_areas, num_heads_comp, num_heads, dff_comp, dff, pe_input)

  return model

def predict(test_file, model_path, summary_dir, batch_size, res_file):
  if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

  dataset_test = tf.data.TFRecordDataset(test_file) \
                    .map(parse_record) \
                    .batch(batch_size).prefetch(2)

  mixed_precision.set_global_policy('mixed_float16')

  model = build_model()
  in_enc_dummy = tf.constant(0., shape=[1, INPUT_LEN, 4])
  model(in_enc_dummy)
  model.load_weights(model_path)

  pred_list = []

  for n, batch in enumerate(dataset_test):
    pred_da, _, _, _ = model(batch['seq'], training=False) # B * (L+16) * 3

    pred = pred.numpy()
    for i in range(pred.shape[0]):
        pred_list.append(pred[i])

  df = pd.DataFrame(data={'pred': pred_list})
  df.to_pickle((os.path.join(summary_dir, res_file)))

  return

def predict_attn(test_file, model_path, summary_dir, batch_size, res_file):
  if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

  dataset_test = tf.data.TFRecordDataset(test_file) \
                    .map(parse_record) \
                    .batch(batch_size).prefetch(2)

  mixed_precision.set_global_policy('mixed_float16')

  model = build_model()
  in_enc_dummy = tf.constant(0., shape=[1, INPUT_LEN, 4])
  model(in_enc_dummy)
  model.load_weights(model_path)

  pred_list = []
  local_list = []
  global_list = []

  for n, batch in enumerate(dataset_test):
    pred, _, local_attn, global_attn_list = model(batch["seq"], training=False) # B * (L+16) * 3
    # local_attn: (batch_size, num_areas, num_heads, l_q, l_q)
    # global_attn_list: (num_layer, batch_size, num_heads, l_q, l_q)

    pred = pred.numpy()
    local_attn = tf.reduce_sum(local_attn, axis=2).numpy() # (batch_size, num_areas, l_q, l_q)
    global_attn = tf.reduce_sum(tf.stack(global_attn_list), axis=[0, 2]).numpy() #(batch_size, l_q, l_q)

    for i in range(pred.shape[0]):
        pred_list.append(pred[i])
        local_list.append(local_attn[i])
        global_list.append(global_attn[i]) 

  df = pd.DataFrame(data={'pred': pred_list, 'local': local_list, 'global':global_list})
  df.to_pickle((os.path.join(summary_dir, res_file)))

  return

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_file', type=str)
  parser.add_argument('--summary_dir', type=str)
  parser.add_argument('--res_file', type=str)
  parser.add_argument('--gpu', type=int, default=None)
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--attention', action='store_true', default=False)
  parser.add_argument('--model_path', type=str, default=None)

  args = parser.parse_args()

  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

  test_file = args.test_file
  batch_size = args.batch_size
  model_path = args.model_path
  
  summary_dir = args.summary_dir
  res_file = args.res_file
  model_path = args.model_path

  if args.attention:
    predict_attn(test_file, model_path, summary_dir, batch_size, res_file)
  else:
    predict(test_file, model_path, summary_dir, batch_size, res_file)

  return

if __name__ == "__main__":
  main()