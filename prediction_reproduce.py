import numpy as np
import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision
import os
import argparse
import pickle
import pandas as pd
from model import SSNet

input_len = 100000

def parse_record(example):    
    context_features = {
        #"id" : tf.io.FixedLenFeature([], dtype=tf.int64),
        "seq" : tf.io.FixedLenFeature([input_len], dtype=tf.int64),
        "da" : tf.io.FixedLenFeature([input_len], dtype=tf.int64),
        "ie" : tf.io.FixedLenFeature([input_len], dtype=tf.int64),
    }

    context_parsed = tf.io.parse_example(serialized=example, features=context_features)
    
    #idx = context_parsed["id"]
    seq = context_parsed["seq"]
    seq_onehot = tf.one_hot(seq, 5)
    mask_n = tf.reduce_all(tf.equal(seq_onehot, 0.), axis=-1, keepdims=True)
    n_token = tf.tile(0.25 * tf.cast(mask_n, tf.float32),[1,4])
    # PAD:[0, 0, 0, 0], A:[1, 0, 0, 0], C:[0, 1, 0, 0,], G:[0, 0, 1, 0], T:[0, 0, 0, 1], N:[0.25, 0.25, 0.25, 0.25]
    in_encoder = seq_onehot[:,1:] + n_token # 100000 * 4

    da = context_parsed["da"] # Donor: 0, Acceptor: 1, None: 2, Pad: 3 100000 * 1

    #PAD:-1, INTRON:0, EXON:1
    ie = tf.expand_dims(tf.cast(context_parsed['ie'], tf.int64), axis=1)
    #PAD:0, INTRON:1, EXON:2
    out_label = ie + 1 # 100000 * 1

    return {"in_encoder":in_encoder, "da": da, "ie": out_label}

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
  pe_input = input_len

  model = SSNet(conv_dim, conv_kernel, num_c_layers, num_t_layers, d_comp, d_model, num_areas, num_heads_comp, num_heads, dff_comp, dff, pe_input)

  return model
 
def parse_record_diff_fix(example):    
    context_features = {
        "ref_seq" : tf.io.FixedLenFeature([input_len], dtype=tf.int64),
        "alt_seq" : tf.io.FixedLenFeature([input_len], dtype=tf.int64),
        "alt_seq2" : tf.io.FixedLenFeature([input_len], dtype=tf.int64),
    }

    context_parsed = tf.io.parse_example(serialized=example, features=context_features)
    
    ref_seq = context_parsed["ref_seq"]
    ref_seq_onehot = tf.one_hot(ref_seq, 5)
    ref_mask_n = tf.reduce_all(tf.equal(ref_seq_onehot, 0.), axis=-1, keepdims=True)
    ref_n_token = tf.tile(0.25 * tf.cast(ref_mask_n, tf.float32),[1,4])
    # PAD:[0, 0, 0, 0], A:[1, 0, 0, 0], C:[0, 1, 0, 0,], G:[0, 0, 1, 0], T:[0, 0, 0, 1], N:[0.25, 0.25, 0.25, 0.25]
    ref_seq_in = ref_seq_onehot[:,1:] + ref_n_token # 100000 * 4

    alt_seq = context_parsed["alt_seq"]
    alt_seq_onehot = tf.one_hot(alt_seq, 5)
    alt_mask_n = tf.reduce_all(tf.equal(alt_seq_onehot, 0.), axis=-1, keepdims=True)
    alt_n_token = tf.tile(0.25 * tf.cast(alt_mask_n, tf.float32),[1,4])
    # PAD:[0, 0, 0, 0], A:[1, 0, 0, 0], C:[0, 1, 0, 0,], G:[0, 0, 1, 0], T:[0, 0, 0, 1], N:[0.25, 0.25, 0.25, 0.25]
    alt_seq_in = alt_seq_onehot[:,1:] + alt_n_token # 100000 * 4

    alt_seq2 = context_parsed["alt_seq2"]
    alt_seq2_onehot = tf.one_hot(alt_seq2, 5)
    alt2_mask_n = tf.reduce_all(tf.equal(alt_seq2_onehot, 0.), axis=-1, keepdims=True)
    alt2_n_token = tf.tile(0.25 * tf.cast(alt2_mask_n, tf.float32),[1,4])
    # PAD:[0, 0, 0, 0], A:[1, 0, 0, 0], C:[0, 1, 0, 0,], G:[0, 0, 1, 0], T:[0, 0, 0, 1], N:[0.25, 0.25, 0.25, 0.25]
    alt_seq2_in = alt_seq2_onehot[:,1:] + alt2_n_token # 100000 * 4

    return {"ref_seq":ref_seq_in, "alt_seq":alt_seq_in, "alt_seq2": alt_seq2_in}


def predict_diff_fix(test_file, model_path, summary_dir, batch_size, start, end, ref_file):
  if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

  dataset_test = tf.data.TFRecordDataset(test_file) \
                    .map(parse_record_diff_fix) \
                    .batch(batch_size).prefetch(2)

  mixed_precision.set_global_policy('mixed_float16')

  model = build_model()
  in_enc_dummy = tf.constant(0., shape=[1, 100000, 4])
  model(in_enc_dummy)
  model.load_weights(model_path)

  if start == 0 and end == sys.maxsize:
    file_name = 'pred_all'
  else:
    file_name = 'pred_' + str(start) + '_' + str(end)
  pickle_file = file_name + '.pickle'
  print("Start:", start)
  print("End:", end, flush=True)

  print(ref_file)
  df_ref = pd.read_pickle(ref_file)

  if 'pos' in df_ref.columns:
    pos_name = 'pos'
  else:
    pos_name = 'hg37_pos'
  if 'ref' in df_ref.columns:
    ref_name = 'ref'
  else:
    ref_name = 'Ref'
  if 'alt' in df_ref.columns:
    alt_name = 'alt'
  else:
    alt_name = 'Alt'

  i = start * batch_size
  pred_ref_list = []
  pred_alt_list = []
  anno_don_list = []
  anno_acc_list = []
  label = []
  pred_no_mask_list = []
  pred_mask_list = []
  for n, batch in enumerate(dataset_test):
    if n < start:
      continue
    elif n > end:
      break
    else:
      if (n+1) % 10 == 0:
        print(n, flush=True)

      pred_ref, _, _, _ = model(batch["ref_seq"], training=False) # B * (L+16) * 3
      pred_alt, _, _, _ = model(batch["alt_seq"], training=False) # B * (L+16) * 3
      pred_alt2, _, _, _ = model(batch["alt_seq2"], training=False)

      for j in range(pred_ref.shape[0]):
        pos = df_ref[pos_name][i]
        tx_s = df_ref['tx_start'][i]
        tx_e = df_ref['tx_end'][i]
        len_ref = len(df_ref[ref_name][i])
        len_alt = len(df_ref[alt_name][i])
        strand = df_ref['strand'][i]
        ref_s = df_ref['ref_s'][i]
        ref_e = df_ref['ref_e'][i]
        alt_e = df_ref['alt_e'][i]
        exon_s = np.array(df_ref['exon_start'][i][1:]).astype(int)
        exon_e = np.array(df_ref['exon_end'][i][:-1]).astype(int)
        exon_e = exon_e[(ref_s <= exon_e) & (ref_e >= exon_e)]
        exon_s = exon_s[(ref_s <= exon_s) & (ref_e >= exon_s)]

        mask_ref = tf.reduce_any(tf.not_equal(batch["ref_seq"][j], 0.), axis=-1)
        mask_alt = tf.reduce_any(tf.not_equal(batch["alt_seq"][j], 0.), axis=-1)
        mask_alt2 = tf.reduce_any(tf.not_equal(batch["alt_seq2"][j], 0.), axis=-1)
        flag_alt2 = tf.reduce_any(mask_alt2, axis=0)

        ref_pred = tf.boolean_mask(pred_ref[j], mask_ref)
        alt_pred = tf.boolean_mask(pred_alt[j], mask_alt)
        alt2_pred = tf.boolean_mask(pred_alt2[j], mask_alt2)
        
        ref_pred = ref_pred.numpy()
        alt_pred = alt_pred.numpy()
        alt2_pred = alt2_pred.numpy()
        ref = ref_pred[:ref_e-ref_s+1,0:2]
        alt = alt_pred[:alt_e-ref_s+1,0:2]

        if len_ref > len_alt:
          if strand == '+':
            if flag_alt2:
              alt2 = alt2_pred[:alt_e-ref_s+1, 0:2]
              pre_mut = alt[:pos-ref_s]
              mut = (alt[pos-ref_s:(pos+len_alt)-ref_s] + alt2[pos-ref_s:(pos+len_alt)-ref_s]) / 2
              post_mut = alt2[(pos+len_alt)-ref_s:]
              alt = np.concatenate([pre_mut, mut, np.zeros((len_ref-len_alt, 2)), post_mut], axis=0)
            else:
              alt = np.concatenate([alt[:(pos+len_alt)-ref_s], np.zeros((len_ref-len_alt, 2)), alt[(pos+len_alt)-ref_s:alt_e-ref_s+1]], axis=0)
            mask_d = np.zeros(len(ref[:,0]), dtype=np.float32)
            mask_d[exon_e-ref_s] = 1.
            mask_a = np.zeros(len(ref[:,1]), dtype=np.float32)
            mask_a[exon_s-ref_s] = 1.
          else:
            if flag_alt2:
              alt2 = alt2_pred[:alt_e-ref_s+1, 0:2]
              pre_mut = alt[:alt_e-(pos+len_alt)+1]
              mut = (alt[alt_e-(pos+len_alt)+1:alt_e-pos+1] + alt2[alt_e-(pos+len_alt)+1:alt_e-pos+1]) / 2
              post_mut = alt2[alt_e-pos+1:]
              alt = np.concatenate([pre_mut, mut, np.zeros((len_ref-len_alt, 2)), post_mut], axis=0)
            else:
              alt = np.concatenate([alt[:alt_e-(pos+len_alt)+1], np.zeros((len_ref-len_alt, 2)), alt[alt_e-(pos+len_alt)+1:alt_e-ref_s+1]], axis=0)
            mask_d = np.zeros(len(ref[:,0]), dtype=np.float32)
            mask_d[ref_e-exon_s] = 1.
            mask_a = np.zeros(len(ref[:,1]), dtype=np.float32)
            mask_a[ref_e-exon_e] = 1.
        elif len_alt > len_ref:
          if strand == '+':
            ref = np.concatenate([ref[:(pos+len_ref)-ref_s], np.zeros((len_alt-len_ref, 2)), ref[(pos+len_ref)-ref_s:ref_e-ref_s+1]], axis=0)
            if flag_alt2:
              alt2 = alt2_pred[:alt_e-ref_s+1-(len_alt-len_ref), 0:2]
              pre_mut = alt[:pos-ref_s]
              if (pos+len_ref-len_alt)-ref_s < 0:
                length = -1 * ((pos+len_ref-len_alt)-ref_s)
                mut_alt = alt[pos-ref_s:(pos+len_alt)-ref_s]
                mut_alt2 = np.concatenate([mut_alt[:length], alt2[0:(pos+len_ref)-ref_s]], axis=0)
                mut = (mut_alt + mut_alt2) / 2
              else:
                mut = (alt[pos-ref_s:(pos+len_alt)-ref_s] + alt2[(pos+len_ref-len_alt)-ref_s:(pos+len_ref)-ref_s]) / 2
              post_mut = alt2[(pos+len_ref)-ref_s:]
              alt = np.concatenate([pre_mut, mut, post_mut], axis=0)
            exon_e = np.where(exon_e > pos+len_ref-1, exon_e+len_alt-len_ref, exon_e)
            exon_s = np.where(exon_s > pos+len_ref-1, exon_s+len_alt-len_ref, exon_s)
            mask_d = np.zeros(len(ref[:,0]), dtype=np.float32)
            mask_d[exon_e-ref_s] = 1.
            mask_a = np.zeros(len(ref[:,1]), dtype=np.float32)
            mask_a[exon_s-ref_s] = 1.
          else:
            ref = np.concatenate([ref[:ref_e-(pos+len_ref)+1], np.zeros((len_alt-len_ref, 2)), ref[ref_e-(pos+len_ref)+1:ref_e-ref_s+1]], axis=0)
            if flag_alt2:
              alt2 = alt2_pred[:alt_e-ref_s+1-(len_alt-len_ref), 0:2]
              pre_mut = alt[:alt_e-(pos+len_alt)+1]
              mut = (alt[alt_e-(pos+len_alt)+1:alt_e-pos+1] +  alt2[alt_e-(pos+len_alt)+1+(len_ref-len_alt):alt_e-pos+1+(len_ref-len_alt)]) / 2
              post_mut = alt2[alt_e-pos+1+len_ref-len_alt:]
              alt = np.concatenate([pre_mut, mut, post_mut], axis=0)
            exon_e = np.where(exon_e > pos+len_ref-1, exon_e+len_alt-len_ref, exon_e)
            exon_s = np.where(exon_s > pos+len_ref-1, exon_s+len_alt-len_ref, exon_s)
            mask_d = np.zeros(len(ref[:,0]), dtype=np.float32)
            mask_d[alt_e-exon_s] = 1.
            mask_a = np.zeros(len(ref[:,1]), dtype=np.float32)
            mask_a[alt_e-exon_e] = 1.
        else:
          if strand == '+':
            mask_d = np.zeros(len(ref[:,0]), dtype=np.float32)
            mask_d[exon_e-ref_s] = 1.
            mask_a = np.zeros(len(ref[:,1]), dtype=np.float32)
            mask_a[exon_s-ref_s] = 1.
          else:
            mask_d = np.zeros(len(ref[:,0]), dtype=np.float32)
            mask_d[ref_e-exon_s] = 1.
            mask_a = np.zeros(len(ref[:,1]), dtype=np.float32)
            mask_a[ref_e-exon_e] = 1.

        pred_ref_list.append(ref)
        pred_alt_list.append(alt)
        anno_don_list.append(mask_d)
        anno_acc_list.append(mask_a)

        # 0:donor 1:acceptor
        diff = alt - ref
        pred_no_mask = 1 if np.max(abs(diff)) > 0.2 else 0
        pred_mask = 1 if np.max(abs(diff[:,0] * (1. - mask_d))) > 0.2 or np.max(abs(diff[:,1] * (1. - mask_a))) > 0.2 else 0

        pred_no_mask_list.append(pred_no_mask)
        pred_mask_list.append(pred_mask)

        label.append(df_ref['label'][i])
        #if df_ref['label'][i] == 'Benign' or df_ref['label'][i] == False:
          #label.append(0)
        #else:
          #label.append(1)
        #if df_ref['label'][i] == False:
          #label.append(0)
        #elif df_ref['label'][i] == True:
          #label.append(1)
        #else:
          #label.append(-1)
        #label.append(df_ref['clinical_significance'][i])
        #if 'benign' in df_ref['Pathogenicity_expert'][i].lower():
          #label.append(0)
        #elif 'pathogenic' in df_ref['Pathogenicity_expert'][i].lower():
            #label.append(1)
        #else:
            #label.append(-1)
        i += 1
        
  print('finish_pred')

  df = pd.DataFrame({'pred_ref': pred_ref_list, 'pred_alt': pred_alt_list, 'anno_don': anno_don_list, 'anno_acc': anno_acc_list, \
                     'label': label, 'res_no_mask': pred_no_mask_list, 'res_mask': pred_mask_list})
  df.to_pickle(os.path.join(summary_dir, pickle_file))

  print('finish pickle')

  return


def predict_diff_sscvdb(test_file, model_path, summary_dir, batch_size, start, end, ref_file):
  if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

  dataset_test = tf.data.TFRecordDataset(test_file) \
                    .map(parse_record_diff) \
                    .batch(batch_size).prefetch(2)

  mixed_precision.set_global_policy('mixed_float16')

  model = build_model()
  in_enc_dummy = tf.constant(0., shape=[1, 100000, 4])
  model(in_enc_dummy)
  model.load_weights(model_path)


  if start == 0 and end == sys.maxsize:
    file_name = 'pred_all'
  else:
    file_name = 'pred_' + str(start) + '_' + str(end)
  pickle_file = file_name + '.pickle'
  print("Start:", start)
  print("End:", end, flush=True)

  print(ref_file)
  df_ref = pd.read_pickle(ref_file)

  if 'pos' in df_ref.columns:
    pos_name = 'pos'
  else:
    pos_name = 'hg37_pos'
  if 'ref' in df_ref.columns:
    ref_name = 'ref'
  else:
    ref_name = 'Ref'
  if 'alt' in df_ref.columns:
    alt_name = 'alt'
  else:
    alt_name = 'Alt'

  i = start * batch_size
  pred_ref_list = []
  pred_alt_list = []
  ref_ss_list = []
  alt_ss_list = []
  alt2_ss_list = []
  pred_no_mask_list = []
  pred_ref_ss_list = []
  pred_alt_ss_list = []
  pred_alt2_ss_list = []
  for n, batch in enumerate(dataset_test):
    if n < start:
      continue
    elif n > end:
      break
    else:
      if (n+1) % 10 == 0:
        print(n, flush=True)

      pred_ref, _, _, _ = model(batch["ref_seq"], training=False) # B * (L+16) * 3
      pred_alt, _, _, _ = model(batch["alt_seq"], training=False) # B * (L+16) * 3

      for j in range(pred_ref.shape[0]):
        pos = df_ref[pos_name][i]
        tx_s = df_ref['tx_start'][i]
        tx_e = df_ref['tx_end'][i]
        strand = df_ref['strand'][i]
        ref_s = df_ref['ref_s'][i]
        ref_e = df_ref['ref_e'][i]
        ss_type = df_ref['Motif_type'][i]
        ref_ss = df_ref['Hijacked_SS_pos'][i]
        alt_ss = df_ref['Primary_SS_pos'][i]
        alt2_ss = df_ref['Secondary_SS_pos'][i]

        mask_ref = tf.reduce_any(tf.not_equal(batch["ref_seq"][j], 0.), axis=-1)
        mask_alt = tf.reduce_any(tf.not_equal(batch["alt_seq"][j], 0.), axis=-1)

        ref_pred = tf.boolean_mask(pred_ref[j], mask_ref)
        alt_pred = tf.boolean_mask(pred_alt[j], mask_alt)
        
        ref_pred = ref_pred.numpy()
        alt_pred = alt_pred.numpy()
        ref = ref_pred[:ref_e-ref_s+1,0:2]
        alt = alt_pred[:ref_e-ref_s+1,0:2]

        if strand == '+':
          if ss_type == 'Donor':
            ref_ss = ref_ss - 1
            alt_ss = alt_ss - 1
            alt2_ss = alt2_ss - 1 if not pd.isna(alt2_ss) else np.nan
          else: # Acceptor
            ref_ss = ref_ss + 1
            alt_ss = alt_ss + 1
            alt2_ss = alt2_ss + 1 if not pd.isna(alt2_ss) else np.nan
        else:
          if ss_type == 'Donor':
            ref_ss = ref_ss + 1
            alt_ss = alt_ss + 1
            alt2_ss = alt2_ss + 1 if not pd.isna(alt2_ss) else np.nan
          else: # Acceptor
            ref_ss = ref_ss - 1
            alt_ss = alt_ss - 1
            alt2_ss = alt2_ss - 1 if not pd.isna(alt2_ss) else np.nan

        ref_ss = ref_ss if (ref_s <= ref_ss) and (ref_e >= ref_ss) else np.nan
        alt_ss = alt_ss if (ref_s <= alt_ss) and (ref_e >= alt_ss) else np.nan
        if not np.isnan(alt2_ss):
          alt2_ss = int(alt2_ss) if (ref_s <= alt2_ss) and (ref_e >= alt2_ss) else np.nan

        if strand == '+':
          mask_ref = np.zeros(len(ref[:,0]), dtype=np.float32)
          if not np.isnan(ref_ss):
            mask_ref[ref_ss-ref_s] = 1.
          mask_alt = np.zeros(len(ref[:,0]), dtype=np.float32)
          if not np.isnan(alt_ss):
            mask_alt[alt_ss-ref_s] = 1.
          mask_alt2 = np.zeros(len(ref[:,0]), dtype=np.float32)
          if not np.isnan(alt2_ss):
            mask_alt2[alt2_ss-ref_s] = 1.
        else:
          mask_ref = np.zeros(len(ref[:,0]), dtype=np.float32)
          if not np.isnan(ref_ss):
            mask_ref[ref_e-ref_ss] = 1.
          mask_alt = np.zeros(len(ref[:,0]), dtype=np.float32)
          if not np.isnan(alt_ss):
            mask_alt[ref_e-alt_ss] = 1.
          mask_alt2 = np.zeros(len(ref[:,0]), dtype=np.float32)
          if not np.isnan(alt2_ss):
            mask_alt2[ref_e-alt2_ss] = 1.

        pred_ref_list.append(ref)
        pred_alt_list.append(alt)
        ref_ss_list.append(mask_ref)
        alt_ss_list.append(mask_alt)
        alt2_ss_list.append(mask_alt2)

        # 0:donor 1:acceptor
        diff = alt - ref
        pred_no_mask = np.max(abs(diff))
        if ss_type == 'Donor':
          pred_ref_ss = np.min(diff[:,0] * mask_ref)
          pred_alt_ss = np.max(diff[:,0] * mask_alt)
          pred_alt2_ss = np.max(diff[:,0] * mask_alt2)
        else: # Acceptor
          pred_ref_ss = np.min(diff[:,1] * mask_ref)
          pred_alt_ss = np.max(diff[:,1] * mask_alt)
          pred_alt2_ss = np.max(diff[:,1] * mask_alt2)

        pred_no_mask_list.append(pred_no_mask)
        pred_ref_ss_list.append(pred_ref_ss)
        pred_alt_ss_list.append(pred_alt_ss)
        pred_alt2_ss_list.append(pred_alt2_ss)

        i += 1
        
  print('finish_pred')

  df = pd.DataFrame({'pred_ref': pred_ref_list, 'pred_alt': pred_alt_list, \
                     'ref_ss': ref_ss_list, 'alt_ss': alt_ss_list, 'alt2_ss': alt2_ss_list, \
                     'res_no_mask': pred_no_mask_list, 'res_ref_ss': pred_ref_ss_list, 'res_alt_ss': pred_alt_ss_list, 'res_alt2_ss': pred_alt2_ss_list})
  df.to_pickle(os.path.join(summary_dir, pickle_file))

  print('finish pickle')

  return

def predict_diff_iravdb(test_file, model_path, summary_dir, batch_size, start, end, ref_file):
  if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

  dataset_test = tf.data.TFRecordDataset(test_file) \
                    .map(parse_record_diff_fix) \
                    .batch(batch_size).prefetch(2)

  mixed_precision.set_global_policy('mixed_float16')

  model = build_model()
  in_enc_dummy = tf.constant(0., shape=[1, 100000, 4])
  model(in_enc_dummy)
  model.load_weights(model_path)

  if start == 0 and end == sys.maxsize:
    file_name = 'pred_all'
  else:
    file_name = 'pred_' + str(start) + '_' + str(end)
  pickle_file = file_name + '.pickle'
  print("Start:", start)
  print("End:", end, flush=True)

  print(ref_file)
  df_ref = pd.read_pickle(ref_file)

  if 'pos' in df_ref.columns:
    pos_name = 'pos'
  else:
    pos_name = 'hg37_pos'
  if 'ref' in df_ref.columns:
    ref_name = 'ref'
  else:
    ref_name = 'Ref'
  if 'alt' in df_ref.columns:
    alt_name = 'alt'
  else:
    alt_name = 'Alt'

  i = start * batch_size
  pred_ref_list = []
  pred_alt_list = []
  target_list = []
  partner_list = []
  pred_no_mask_list = []
  pred_target_list = []
  pred_partner_list = []

  for n, batch in enumerate(dataset_test):
    if n < start:
      continue
    elif n > end:
      break
    else:
      if (n+1) % 10 == 0:
        print(n, flush=True)

      pred_ref, _, _, _ = model(batch["ref_seq"], training=False) # B * (L+16) * 3
      pred_alt, _, _, _ = model(batch["alt_seq"], training=False) # B * (L+16) * 3
      pred_alt2, _, _, _ = model(batch["alt_seq2"], training=False)

      for j in range(pred_ref.shape[0]):
        pos = df_ref[pos_name][i]
        tx_s = df_ref['tx_start'][i]
        tx_e = df_ref['tx_end'][i]
        strand = df_ref['strand'][i]
        len_ref = len(df_ref[ref_name][i])
        len_alt = len(df_ref[alt_name][i])
        ref_s = df_ref['ref_s'][i]
        ref_e = df_ref['ref_e'][i]
        alt_e = df_ref['alt_e'][i]
        ss_type = df_ref['Motif_Type'][i]
        target_pos = df_ref['target_pos'][i]
        partner_pos = df_ref['partner_pos'][i]

        mask_ref = tf.reduce_any(tf.not_equal(batch["ref_seq"][j], 0.), axis=-1)
        mask_alt = tf.reduce_any(tf.not_equal(batch["alt_seq"][j], 0.), axis=-1)
        mask_alt2 = tf.reduce_any(tf.not_equal(batch["alt_seq2"][j], 0.), axis=-1)
        flag_alt2 = tf.reduce_any(mask_alt2, axis=0)

        ref_pred = tf.boolean_mask(pred_ref[j], mask_ref)
        alt_pred = tf.boolean_mask(pred_alt[j], mask_alt)
        alt2_pred = tf.boolean_mask(pred_alt2[j], mask_alt2)
        
        ref_pred = ref_pred.numpy()
        alt_pred = alt_pred.numpy()
        alt2_pred = alt2_pred.numpy()
        ref = ref_pred[:ref_e-ref_s+1,0:2]
        alt = alt_pred[:ref_e-ref_s+1,0:2]

        target_pos = target_pos if (ref_s <= target_pos) and (ref_e >= target_pos) else np.nan
        partner_pos = partner_pos if (ref_s <= partner_pos) and (ref_e >= partner_pos) else np.nan

        if len_ref > len_alt:
          if strand == '+':
            if flag_alt2:
              alt2 = alt2_pred[:alt_e-ref_s+1, 0:2]
              pre_mut = alt[:pos-ref_s]
              mut = (alt[pos-ref_s:(pos+len_alt)-ref_s] + alt2[pos-ref_s:(pos+len_alt)-ref_s]) / 2
              post_mut = alt2[(pos+len_alt)-ref_s:]
              alt = np.concatenate([pre_mut, mut, np.zeros((len_ref-len_alt, 2)), post_mut], axis=0)
            else:
              alt = np.concatenate([alt[:(pos+len_alt)-ref_s], np.zeros((len_ref-len_alt, 2)), alt[(pos+len_alt)-ref_s:alt_e-ref_s+1]], axis=0)
            mask_target = np.zeros(len(ref[:,0]), dtype=np.float32)
            if not np.isnan(target_pos):
              mask_target[target_pos - ref_s] = 1.
            mask_partner = np.zeros(len(ref[:,0]), dtype=np.float32)
            if not np.isnan(partner_pos):
              mask_partner[partner_pos - ref_s] = 1.
          else:
            if flag_alt2:
              alt2 = alt2_pred[:alt_e-ref_s+1, 0:2]
              pre_mut = alt[:alt_e-(pos+len_alt)+1]
              mut = (alt[alt_e-(pos+len_alt)+1:alt_e-pos+1] + alt2[alt_e-(pos+len_alt)+1:alt_e-pos+1]) / 2
              post_mut = alt2[alt_e-pos+1:]
              alt = np.concatenate([pre_mut, mut, np.zeros((len_ref-len_alt, 2)), post_mut], axis=0)
            else:
              alt = np.concatenate([alt[:alt_e-(pos+len_alt)+1], np.zeros((len_ref-len_alt, 2)), alt[alt_e-(pos+len_alt)+1:alt_e-ref_s+1]], axis=0)
            mask_target = np.zeros(len(ref[:,0]), dtype=np.float32)
            if not np.isnan(target_pos):
              mask_target[ref_e - target_pos] = 1.
            mask_partner = np.zeros(len(ref[:,0]), dtype=np.float32)
            if not np.isnan(partner_pos):
              mask_partner[ref_e - partner_pos] = 1.
        elif len_alt > len_ref:
          if strand == '+':
            ref = np.concatenate([ref[:(pos+len_ref)-ref_s], np.zeros((len_alt-len_ref, 2)), ref[(pos+len_ref)-ref_s:ref_e-ref_s+1]], axis=0)
            if flag_alt2:
              alt2 = alt2_pred[:alt_e-ref_s+1-(len_alt-len_ref), 0:2]
              pre_mut = alt[:pos-ref_s]
              mut = (alt[pos-ref_s:(pos+len_alt)-ref_s] + alt2[(pos+len_ref-len_alt)-ref_s:(pos+len_ref)-ref_s]) / 2
              post_mut = alt2[(pos+len_ref)-ref_s:]
              alt = np.concatenate([pre_mut, mut, post_mut], axis=0)
            mask_target = np.zeros(len(ref[:,0]), dtype=np.float32)
            if not np.isnan(target_pos):
              target_pos = target_pos+len_alt-len_ref if target_pos > pos+len_ref-1 else target_pos
              mask_target[target_pos - ref_s] = 1.
            mask_partner = np.zeros(len(ref[:,0]), dtype=np.float32)
            if not np.isnan(partner_pos):
              partner_pos = partner_pos+len_alt-len_ref if partner_pos > pos+len_ref-1 else partner_pos
              mask_partner[partner_pos - ref_s] = 1.
          else:
            ref = np.concatenate([ref[:ref_e-(pos+len_ref)+1], np.zeros((len_alt-len_ref, 2)), ref[ref_e-(pos+len_ref)+1:ref_e-ref_s+1]], axis=0)
            if flag_alt2:
              alt2 = alt2_pred[:alt_e-ref_s+1-(len_alt-len_ref), 0:2]
              pre_mut = alt[:alt_e-(pos+len_alt)+1]
              mut = (alt[alt_e-(pos+len_alt)+1:alt_e-pos+1] +  alt2[alt_e-(pos+len_alt)+1+(len_ref-len_alt):alt_e-pos+1+(len_ref-len_alt)]) / 2
              post_mut = alt2[alt_e-pos+1+len_ref-len_alt:]
              alt = np.concatenate([pre_mut, mut, post_mut], axis=0)
            mask_target = np.zeros(len(ref[:,0]), dtype=np.float32)
            if not np.isnan(target_pos):
              target_pos = target_pos+len_alt-len_ref if target_pos > pos+len_ref-1 else target_pos
              mask_target[alt_e - target_pos] = 1.
            mask_partner = np.zeros(len(ref[:,0]), dtype=np.float32)
            if not np.isnan(partner_pos):
              partner_pos = partner_pos+len_alt-len_ref if partner_pos > pos+len_ref-1 else partner_pos
              mask_partner[alt_e - partner_pos] = 1.
        else:
          if strand == '+':
            mask_target = np.zeros(len(ref[:,0]), dtype=np.float32)
            if not np.isnan(target_pos):
              mask_target[target_pos - ref_s] = 1.
            mask_partner = np.zeros(len(ref[:,0]), dtype=np.float32)
            if not np.isnan(partner_pos):
              mask_partner[partner_pos - ref_s] = 1.
          else:
            mask_target = np.zeros(len(ref[:,0]), dtype=np.float32)
            if not np.isnan(target_pos):
              mask_target[ref_e - target_pos] = 1.
            mask_partner = np.zeros(len(ref[:,0]), dtype=np.float32)
            if not np.isnan(partner_pos):
              mask_partner[ref_e - partner_pos] = 1.

        pred_ref_list.append(ref)
        pred_alt_list.append(alt)
        target_list.append(mask_target)
        partner_list.append(mask_partner)

        # 0:donor 1:acceptor
        diff = alt - ref
        pred_no_mask = np.max(abs(diff))
        if ss_type == 'donor':
          pred_target = np.min(diff[:,0] * mask_target)
          pred_partner = np.min(diff[:,1] * mask_partner)
        else: # Acceptor
          pred_target = np.min(diff[:,1] * mask_target)
          pred_partner = np.min(diff[:,0] * mask_partner)

        pred_no_mask_list.append(pred_no_mask)
        pred_target_list.append(pred_target)
        pred_partner_list.append(pred_partner)

        i += 1
          
  print('finish_pred')

  df = pd.DataFrame({'pred_ref': pred_ref_list, 'pred_alt': pred_alt_list, \
                     'target_pos': target_list, 'partner_pos': partner_list, \
                     'res_no_mask': pred_no_mask_list, 'res_target': pred_target_list, 'res_partner': pred_partner_list})
  df.to_pickle(os.path.join(summary_dir, pickle_file))

  print('finish pickle')

  return

def predict_DMD(test_file, model_path, summary_dir, batch_size, start, end, summary_name=""):
  if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

  dataset_test = pd.read_csv(test_file)

  mixed_precision.set_global_policy('mixed_float16')

  model = build_model()
  in_enc_dummy = tf.constant(0., shape=[1, 100000, 4])
  model(in_enc_dummy)
  model.load_weights(model_path)

  file_name = 'pred'
  if summary_name != "":
    file_name +='_' + summary_name
  pickle_file = file_name + '.pickle'
  print("Start:", start)
  print("End:", end, flush=True)

  pred_da_list = []

  ref_pred_don = None
  tmp_dist = None
  tmp_max = None
  res_dict = {}

  for i in range(len(dataset_test)):
    seq = dataset_test['seq'][i]
    seq = seq.replace('A', '1,').replace('C', '2,').replace('G', '3,').replace('T', '4,')
    seq = re.sub(r'[A-Z]{1}', '-1,', seq)
    seq = [int(x) for x in seq.split(',')[:-1]]
    seq = tf.constant(seq, dtype=tf.int64)
    seq_onehot = tf.one_hot(seq, 5)
    mask_n = tf.reduce_all(tf.equal(seq_onehot, 0.), axis=-1, keepdims=True)
    n_token = tf.tile(0.25 * tf.cast(mask_n, tf.float32),[1,4])
    in_encoder = seq_onehot[:,1:] + n_token # 100000 * 4
    in_encoder = tf.expand_dims(in_encoder, axis=0) # 1 * 100000 * 4

    _, L, _ = in_encoder.shape
    pad_len = input_len - L

    if pad_len > 0:
        paddings = tf.constant([[0,0], [0,pad_len], [0,0]])
        in_encoder_padded = tf.pad(in_encoder, paddings, constant_values=0.0)
    else:
        in_encoder_padded = in_encoder

    pred_da, _, _, _ = model(in_encoder_padded, training=False) # B * (L+16) * 3
    # local_attn: (batch_size, num_areas, num_heads, l_q, l_q)
    # global_attn_list: (num_layer, batch_size, num_heads, l_q, l_q)

    da_pred = pred_da.numpy()
    da_pred = da_pred[0, :L, :]
    pred_da_list.append(da_pred)

    dist = dataset_test['distance'][i]
    shift = dataset_test['shift'][i]
    
    if i == 0:
      ref_pred_don = da_pred[4999, 0]
      print(dist, shift, ref_pred_don, 0)
    else:
      pred_diff = ref_pred_don - da_pred[4999,0]
      print(dist, shift, da_pred[4999,0], pred_diff)

    if dist != tmp_dist:
      if tmp_max is not None:
        res_dict[tmp_dist] = tmp_max
      if i != 0:
        tmp_max = ref_pred_don - da_pred[4999,0]
      tmp_dist = dist
    else:
      tmp_max = max(tmp_max, ref_pred_don - da_pred[4999,0])

  res_dict[tmp_dist] = tmp_max

  print("")
  for key in res_dict:
    print(key, res_dict[key])
  print("")
    
  print('finish_pred')

  df = pd.DataFrame(data={'pred': pred_da_list})
  df.to_pickle((os.path.join(summary_dir, pickle_file)))

  print('finish pickle')


  return

def predict_diff_clinvar(test_file, model_path, summary_dir, batch_size, start, end, ref_file):
  if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

  dataset_test = tf.data.TFRecordDataset(test_file) \
                    .map(parse_record_diff) \
                    .batch(batch_size).prefetch(2)

  mixed_precision.set_global_policy('mixed_float16')

  model = build_model()
  in_enc_dummy = tf.constant(0., shape=[1, 100000, 4])
  model(in_enc_dummy)
  model.load_weights(model_path)

  if start == 0 and end == sys.maxsize:
    file_name = 'pred_all'
  else:
    file_name = 'pred_' + str(start) + '_' + str(end)
  pickle_file = file_name + '.pickle'
  print("Start:", start)
  print("End:", end, flush=True)

  print(ref_file)
  df_ref = pd.read_pickle(ref_file)

  if 'pos' in df_ref.columns:
    pos_name = 'pos'
  else:
    pos_name = 'hg37_pos'
  if 'ref' in df_ref.columns:
    ref_name = 'ref'
  else:
    ref_name = 'Ref'
  if 'alt' in df_ref.columns:
    alt_name = 'alt'
  else:
    alt_name = 'Alt'

  i = start * batch_size
  pred_ref_list = []
  pred_alt_list = []

  dist_list = []

  donor_list = []
  acceptor_list = []

  pred_no_mask_list = []
  pred_donor_list = []
  pred_acceptor_list = []
  pred_max_list = []

  for n, batch in enumerate(dataset_test):
    if n < start:
      continue
    elif n > end:
      break
    else:
      if (n+1) % 10 == 0:
        print(n, flush=True)

      pred_ref, _, _, _ = model(batch["ref_seq"], training=False) # B * (L+16) * 3
      pred_alt, _, _, _ = model(batch["alt_seq"], training=False) # B * (L+16) * 3

      for j in range(pred_ref.shape[0]):
        pos = df_ref[pos_name][i]
        tx_s = df_ref['tx_start'][i]
        tx_e = df_ref['tx_end'][i]
        strand = df_ref['strand'][i]
        ref_s = df_ref['ref_s'][i]
        ref_e = df_ref['ref_e'][i]

        donor_pos = df_ref['donor'][i]
        acceptor_pos = df_ref['acceptor'][i]

        dist_list.append(min(abs(pos-donor_pos), abs(pos-acceptor_pos)))

        mask_ref = tf.reduce_any(tf.not_equal(batch["ref_seq"][j], 0.), axis=-1)
        mask_alt = tf.reduce_any(tf.not_equal(batch["alt_seq"][j], 0.), axis=-1)

        ref_pred = tf.boolean_mask(pred_ref[j], mask_ref)
        alt_pred = tf.boolean_mask(pred_alt[j], mask_alt)
        
        ref_pred = ref_pred.numpy()
        alt_pred = alt_pred.numpy()
        ref = ref_pred[:ref_e-ref_s+1,0:2]
        alt = alt_pred[:ref_e-ref_s+1,0:2]

        donor_pos = int(donor_pos) if (ref_s <= donor_pos) and (ref_e >= donor_pos) else np.nan
        acceptor_pos = int(acceptor_pos) if (ref_s <= acceptor_pos) and (ref_e >= acceptor_pos) else np.nan

        if strand == '+':
          mask_don = np.zeros(len(ref[:,0]), dtype=np.float32)
          if not np.isnan(donor_pos):
            mask_don[donor_pos-ref_s] = 1.
          mask_acc = np.zeros(len(ref[:,1]), dtype=np.float32)
          if not np.isnan(acceptor_pos):
            mask_acc[acceptor_pos-ref_s] = 1.
        else:
          mask_don = np.zeros(len(ref[:,0]), dtype=np.float32)
          if not np.isnan(donor_pos):
            mask_don[ref_e-donor_pos] = 1.
          mask_acc = np.zeros(len(ref[:,1]), dtype=np.float32)
          if not np.isnan(acceptor_pos):
            mask_acc[ref_e-acceptor_pos] = 1.

        pred_ref_list.append(ref)
        pred_alt_list.append(alt)
        donor_list.append(mask_don)
        acceptor_list.append(mask_acc)

        # 0:donor 1:acceptor
        diff = alt - ref
        pred_no_mask = np.max(abs(diff))

        pred_donor = np.max(np.abs(diff[:,0] * mask_don))
        pred_acceptor = np.max(np.abs(diff[:,1] * mask_acc))

        pred_no_mask_list.append(pred_no_mask)
        pred_donor_list.append(pred_donor)
        pred_acceptor_list.append(pred_acceptor)
        pred_max_list.append(max(pred_donor, pred_acceptor))

        i += 1
        
  print('finish_pred')

  print("")
  for (dist, pred_max) in zip(dist_list, pred_max_list):
    print(str(dist) + '\t' + str(pred_max))
  print("")

  df = pd.DataFrame({'pred_ref': pred_ref_list, 'pred_alt': pred_alt_list, \
                     'donor_pos': donor_list, 'acceptor_pos': acceptor_list, 'dist': dist_list, \
                     'res_no_mask': pred_no_mask_list, 'res_donor': pred_donor_list, 'res_acceptor': pred_acceptor_list, 'res_max': pred_max_list})
  df.to_pickle(os.path.join(summary_dir, pickle_file))

  print('finish pickle')

  return

def predict_attn_diff(test_file, model_path, summary_dir, batch_size, start, end):
  if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

  dataset_test = tf.data.TFRecordDataset(test_file) \
                    .map(parse_record_diff_fix) \
                    .batch(batch_size).prefetch(2)

  mixed_precision.set_global_policy('mixed_float16')

  model = build_model()
  in_enc_dummy = tf.constant(0., shape=[1, 100000, 4])
  model(in_enc_dummy)
  model.load_weights(model_path)

  if start == 0 and end == sys.maxsize:
    file_name = 'pred_attn_all'
  else:
    file_name = 'pred_attn_' + str(start) + '_' + str(end)
  pickle_file = file_name + '.pickle'
  print("Start:", start)
  print("End:", end, flush=True)

  i = start * batch_size
  local_list1 = []
  global_list1 = []
  local_list2 = []
  global_list2 = []
  local_list3 = []
  global_list3 = []
  for n, batch in enumerate(dataset_test):
    if n < start:
      continue
    elif n > end:
      break
    else:
      if (n+1) % 10 == 0:
        print(n, flush=True)

      # local_attn: (batch_size, num_areas, num_heads, l_q, l_q)
      # global_attn_list: (num_layer, batch_size, num_heads, l_q, l_q)
      _, _, local_attn1, global_attn_list1 = model(batch["ref_seq"], training=False) # B * (L+16) * 3
      _, _, local_attn2, global_attn_list2 = model(batch["alt_seq"], training=False) # B * (L+16) * 3
      _, _, local_attn3, global_attn_list3 = model(batch["alt_seq2"], training=False)

      local_attn1 = tf.reduce_sum(local_attn1, axis=[2, 3]).numpy() # (batch_size, num_areas, l_q, l_q)
      global_attn1 = tf.reduce_sum(tf.stack(global_attn_list1), axis=[0, 2]).numpy() #(batch_size, l_q, l_q)
      local_attn2 = tf.reduce_sum(local_attn2, axis=[2, 3]).numpy() # (batch_size, num_areas, l_q, l_q)
      global_attn2 = tf.reduce_sum(tf.stack(global_attn_list2), axis=[0, 2]).numpy() #(batch_size, l_q, l_q)
      local_attn3 = tf.reduce_sum(local_attn3, axis=[2, 3]).numpy() # (batch_size, num_areas, l_q, l_q)
      global_attn3 = tf.reduce_sum(tf.stack(global_attn_list3), axis=[0, 2]).numpy() #(batch_size, l_q, l_q)

      for j in range(batch_size):
        local_list1.append(local_attn1[j])
        global_list1.append(global_attn1[j])
        local_list2.append(local_attn2[j])
        global_list2.append(global_attn2[j]) 
        local_list3.append(local_attn3[j])
        global_list3.append(global_attn3[j]) 

        i += 1
        
  print('finish_pred')

  df = pd.DataFrame(data={'local_ref': local_list1, 'global_ref':global_list1, \
                          'local_alt': local_list2, 'global_alt':global_list2, \
                          'local_alt2': local_list3, 'global_alt2':global_list3,})
  df.to_pickle(os.path.join(summary_dir, pickle_file))

  print('finish pickle')

  return

def predict_attn_ESE(test_file, model_path, summary_dir, batch_size, start, end, summary_name=""):
  if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

  dataset_test = pd.read_csv(test_file)

  mixed_precision.set_global_policy('mixed_float16')

  model = build_model()
  in_enc_dummy = tf.constant(0., shape=[1, 100000, 4])
  model(in_enc_dummy)
  model.load_weights(model_path)

  file_name = 'pred_attn'
  if summary_name != "":
    file_name +='_' + summary_name
  pickle_file = file_name + '.pickle'
  print("Start:", start)
  print("End:", end, flush=True)

  pred_da_list = []
  local_attn_list = []
  global_attn_list = []

  for i in range(len(dataset_test)):
    seq = dataset_test['seq'][i]
    seq = seq.replace('A', '1,').replace('C', '2,').replace('G', '3,').replace('T', '4,')
    seq = re.sub(r'[A-Z]{1}', '-1,', seq)
    seq = [int(x) for x in seq.split(',')[:-1]]
    seq = tf.constant(seq, dtype=tf.int64)
    seq_onehot = tf.one_hot(seq, 5)
    mask_n = tf.reduce_all(tf.equal(seq_onehot, 0.), axis=-1, keepdims=True)
    n_token = tf.tile(0.25 * tf.cast(mask_n, tf.float32),[1,4])
    in_encoder = seq_onehot[:,1:] + n_token # 100000 * 4
    in_encoder = tf.expand_dims(in_encoder, axis=0) # 1 * 100000 * 4

    _, L, _ = in_encoder.shape
    pad_len = input_len - L

    if pad_len > 0:
        paddings = tf.constant([[0,0], [0,pad_len], [0,0]])
        in_encoder_padded = tf.pad(in_encoder, paddings, constant_values=0.0)
    else:
        in_encoder_padded = in_encoder

    pred_da, _, local_attn, global_attn = model(in_encoder_padded, training=False) # B * (L+16) * 3
    # local_attn: (batch_size, num_areas, num_heads, l_q, l_q)
    # global_attn_list: (num_layer, batch_size, num_heads, l_q, l_q)

    da_pred = pred_da.numpy()
    pred_da_list.append(da_pred[0])

    local_attn = tf.reduce_sum(local_attn, axis=2)
    local_attn = local_attn.numpy() # (batch_size, num_areas, l_q, l_q)
    local_attn_list.append(local_attn[0])

    global_attn = tf.stack(global_attn)
    global_attn = tf.reduce_sum(global_attn, axis=[0, 2])
    global_attn = global_attn.numpy() #(batch_size, l_q, l_q)
    global_attn_list.append(global_attn[0])
    
  print('finish_pred')

  df = pd.DataFrame(data={'local': local_attn_list, 'global':global_attn_list, 'pred': pred_da_list})
  df.to_pickle((os.path.join(summary_dir, pickle_file)))

  print('finish pickle')

  return

def predict_attn_mask_diff(test_file, model_path, summary_dir, batch_size, start, end):
  if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

  mixed_precision.set_global_policy('mixed_float16')

  model = build_model()
  in_enc_dummy = tf.constant(0., shape=[1, 100000, 4])
  model(in_enc_dummy)
  model.load_weights(model_path)

  file_name = 'pred_attn_mask' #+ '_' + str(start) + '_' + str(end)
  pickle_file = file_name + '.pickle'
  print("Start:", start)
  print("End:", end, flush=True)

  dataset_test = pd.read_pickle(test_file)

  tmp_id = None
  tmp_ref_pred = None
  pred_list = []
  diff_list = []
  id_list = []
  high_low_list = []

  for i in range(len(dataset_test)):
    seq = dataset_test['seq'][i]
    seq = tf.constant(seq, dtype=tf.int64)
    mask_seq = tf.not_equal(seq, 0)
    seq_onehot = tf.one_hot(seq, 5)
    mask_n = tf.reduce_all(tf.equal(seq_onehot, 0.), axis=-1, keepdims=True)
    n_token = tf.tile(0.25 * tf.cast(mask_n, tf.float32),[1,4])
    in_encoder = seq_onehot[:,1:] + n_token # 100000 * 4
    in_encoder = tf.expand_dims(in_encoder, axis=0) # 1 * 100000 * 4

    pred_da, _, _,_ = model(in_encoder, training=False) # B * (L+16) * 3

    pred_da = tf.boolean_mask(pred_da[0], mask_seq)
    da_pred = pred_da.numpy()

    pred_list.append(da_pred)

    idx = dataset_test['id'][i]
    label = dataset_test['label'][i]

    if idx != tmp_id:
      if label == 0:
        tmp_ref_pred = da_pred
        tmp_id = idx
      else:
        print("Error: First entry should be reference.")
    else:
      pred_diff = np.max(np.abs(da_pred - tmp_ref_pred))
      diff_list.append(pred_diff)
      id_list.append(idx)
      if label == -1:
        high_low_list.append('low')
      elif label == 1:
        high_low_list.append('high')
    
  print('finish_pred')

  df = pd.DataFrame({'pred': pred_list})
  df.to_pickle((os.path.join(summary_dir, pickle_file)))

  print('finish pickle')

  df_res = pd.DataFrame({'id': id_list, 'diff': diff_list, 'high_low': high_low_list})
  df_res.to_csv('pred_diff.tsv', index=False, sep='\t')

  print('finish tsv')

  stats = df_res.groupby('high_low')['diff'].describe()
  print(stats[['count', 'mean', 'std', 'min', 'max']])

  return



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cv_splits_file', type=str)
  parser.add_argument('--data_dir', type=str)
  parser.add_argument('--summary_dir', type=str)
  parser.add_argument('--cv', type=int)
  parser.add_argument('--gpu', type=int, default=None)
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--model_save_step', type=int, default=400)
  parser.add_argument('--num_epoch', type=int, default=20)
  parser.add_argument('--model_path', type=str, default=None)
  parser.add_argument('--start', type=int, default=0)
  parser.add_argument('--end', type=int, default=sys.maxsize)
  parser.add_argument('--data_type', type=str)
  parser.add_argument('--pred_ref', type=str)


  args = parser.parse_args()

  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

  cv_file = None
  if args.cv_splits_file:
      with open(args.cv_splits_file, 'r') as f:
          cv_file = yaml.safe_load(f)

  cv = args.cv
  data_dir = args.data_dir
  batch_size = args.batch_size
  summary_step = args.summary_step
  model_path = args.model_path

  start = args.start
  end = args.end

  data_type = args.data_type

  if data_type == 'SpliceVarDB' or data_type == 'BRCA':
    test_files = [os.path.join(data_dir, f) for f in cv_file[cv-1]['test']]
    summary_dir = os.path.join(args.summary_dir, str(cv))
    model_path = args.model_path
    predict_diff_fix(test_files, model_path, summary_dir, batch_size, start, end, args.pred_ref)
    return

  elif data_type == 'sscvdb':
    test_files = [os.path.join(data_dir, f) for f in cv_file[cv-1]['test']]
    summary_dir = os.path.join(args.summary_dir, str(cv))
    model_path = args.model_path
    predict_diff_sscvdb(test_files, model_path, summary_dir, batch_size, start, end, args.pred_ref)
    return

  elif data_type == 'iravdb':
    test_files = [os.path.join(data_dir, f) for f in cv_file[cv-1]['test']]
    summary_dir = os.path.join(args.summary_dir, str(cv))
    model_path = args.model_path
    predict_diff_iravdb(test_files, model_path, summary_dir, batch_size, start, end, args.pred_ref)
    return

  elif data_type == 'DMD':
    test_files = os.path.join(data_dir, 'DMD_seq_ie.csv')
    summary_dir = os.path.join(args.summary_dir)
    model_path = args.model_path
    predict_DMD(test_files, model_path, summary_dir, batch_size, start, end)
    return

  elif data_type == 'clinvar':
    test_files = [os.path.join(data_dir, f) for f in cv_file[cv-1]['test']]
    summary_dir = os.path.join(args.summary_dir, str(cv))
    model_path = args.model_path
    predict_diff_clin_dist(test_files, model_path, summary_dir, batch_size, start, end, args.pred_ref)
    return

  elif data_type == 'IgM':
    test_files = os.path.join(data_dir, 'IgM_data_seq_ie.csv')
    summary_dir = os.path.join(args.summary_dir)
    model_path = args.model_path
    predict_attn_ESE(test_files, model_path, summary_dir, batch_size, start, end, summary_name="IgM")
    return

  elif data_type == 'FAS':
    test_files = os.path.join(data_dir, 'FAS_data_seq_ie.csv')
    summary_dir = os.path.join(args.summary_dir)
    model_path = args.model_path
    predict_attn_ESE(test_files, model_path, summary_dir, batch_size, start, end, summary_name="FAS")
    return
    
  elif data_type == 'attn_insilico':
    test_files = os.path.join(data_dir, 'mask_motif_dataset.pickle')
    summary_dir = args.summary_dir
    model_path = args.model_path
    predict_attn_mask_diff(test_files, model_path, summary_dir, batch_size, start, end)
    return

  elif data_type == 'BRCA_attn':
    test_files = [os.path.join(data_dir, f) for f in cv_file[cv-1]['test']]
    summary_dir = os.path.join(args.summary_dir, str(cv))
    model_path = args.model_path
    predict_attn_diff(test_files, model_path, summary_dir, batch_size, start, end)
    return

if __name__ == "__main__":
  main()