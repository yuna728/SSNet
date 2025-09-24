import numpy as np
import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision
import os
import argparse
import pickle
import pandas as pd
from cyvcf2 import VCF
from Bio import SeqIO
import re
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

  print(f"Start prediction of {test_file}...", flush=True)
  for n, batch in enumerate(dataset_test):
    pred, _, _, _ = model(batch['seq'], training=False) # B * (L+16) * 3

    pred = pred.numpy()
    for i in range(pred.shape[0]):
        pred_list.append(pred[i])

  df = pd.DataFrame(data={'pred': pred_list})
  df.to_pickle((os.path.join(summary_dir, res_file)))
  print(f"Results saved to {res_file}", flush=True)

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

  print(f"Start prediction of {test_file}...", flush=True)
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

  print(f"Results and attention saved to {res_file}", flush=True)

  return


def load_annotation(annotation_path):
    df = pd.read_csv(annotation_path, sep="\t", comment="#",
                     names=["NAME", "CHROM", "STRAND", "TX_START", "TX_END", "EXON_START", "EXON_END"])
    df["EXON_START"] = df["EXON_START"].apply(lambda x: [int(v)+1 for v in str(x).split(",")[:-1] if v])
    df["EXON_END"]   = df["EXON_END"].apply(lambda x: [int(v) for v in str(x).split(",")[:-1] if v])
    return df

def match_variant_to_annotation(vcf_path, annotation_path):
    ann_df = load_annotation(annotation_path)
    vcf = VCF(vcf_path)
    results = []

    for var in vcf:
        chrom, pos = var.CHROM, var.POS
        if chrom.startswith('chr'):
            chrom = chrom[3:]  # chr1, chr2 形式に変換

        matches = ann_df[(ann_df["CHROM"] == chrom) & 
                         (ann_df["TX_START"] <= pos) & (ann_df["TX_END"] >= pos)]
        if matches.empty:
            print(f"No annotation found for variant {chrom}:{pos}")
            continue

        for _, row in matches.iterrows():
            results.append({
                "chrom": chrom,
                "pos": pos,
                "ref": var.REF,
                "alt": var.ALT[0],
                "gene": row["NAME"],
                "strand": row["STRAND"],
                "tx_start": row["TX_START"]+1,
                "tx_end": row["TX_END"],
                "exon_starts": row["EXON_START"],
                "exon_ends": row["EXON_END"],
            })

    return pd.DataFrame(results)

def add_sequence(df, fasta_dir):
  df["orig_order"] = range(len(df))

  df = df.sort_values("chrom").reset_index(drop=True)

  tmp_chr = None
  df['ref_seq'] = ''
  df['alt_seq'] = ''

  for i in range(len(df)):
      if df['chrom'][i] != tmp_chr:
        tmp_chr = df['chrom'][i]
        fasta_file = os.path.join(fasta_dir,'chr' + str(tmp_chr) + '.fa')
        input_seq = SeqIO.parse(fasta_file, format="fasta")
        for record in input_seq:
            ref_seq = str(record.seq)
        print(f"Processing chromosome: chr{tmp_chr}", flush=True)

      if df['strand'][i] == '+':
          df['ref_seq'][i] = ref_seq[df['tx_start'][i]-1:df['tx_end'][i]].upper()
          assert ref_seq[df['pos'][i]-1:df['pos'][i]-1+len(df['ref'][i])].upper() == df['ref'][i].upper(), \
                 f"Assertion failed: chrom={df['chrom'][i]}, pos={df['pos'][i]}, ref_vcf={df['ref'][i]}, ref_fasta={ref_seq[df['pos'][i]-1:df['pos'][i]-1+len(df['ref'][i])]}"

          df['alt_seq'][i] = ref_seq[df['tx_start'][i]-1:df['pos'][i]-1].upper() + df['alt'][i].upper() + ref_seq[df['pos'][i]-1+len(df['ref'][i]):df['tx_end'][i]].upper()
      else:
          tmp_ref_str = ref_seq[df['tx_start'][i]-1:df['tx_end'][i]].upper()
          assert ref_seq[df['pos'][i]-1:df['pos'][i]-1+len(df['ref'][i])].upper() == df['ref'][i].upper(), \
                  f"Assertion failed: chrom={df['chrom'][i]}, pos={df['pos'][i]}, ref_vcf={df['ref'][i]}, ref_fasta={ref_seq[df['pos'][i]-1:df['pos'][i]-1+len(df['ref'][i])]}"
          tmp_alt_str = ref_seq[df['tx_start'][i]-1:df['pos'][i]-1].upper() + df['alt'][i].upper() + ref_seq[df['pos'][i]-1+len(df['ref'][i]):df['tx_end'][i]].upper()
          df['ref_seq'][i] = tmp_ref_str[::-1]
          df['alt_seq'][i] = tmp_alt_str[::-1]

  df = df.sort_values("orig_order").drop(columns=["orig_order"]).reset_index(drop=True)

  def change_seq(seq):
    translation_table = str.maketrans('ACGTN', 'TGCAN')
    new_seq = seq.translate(translation_table)
    assert len(seq) == len(new_seq)
    return seq.translate(translation_table)

  def fix_ref_seq(df):
      seq = df['ref_seq'].upper()
      if df['strand'] == '-':
          seq = change_seq(seq)
      return seq

  def fix_alt_seq(df):
      seq = df['alt_seq'].upper()
      if df['strand'] == '-':
          seq = change_seq(seq)
      return seq

  df['ref_seq'] = df.apply(fix_ref_seq, axis=1)
  df['alt_seq'] = df.apply(fix_alt_seq, axis=1)

  return df

def pad_sequence(df):
  def seq2int(seq):
    seq = seq.replace('A', '1,').replace('C', '2,').replace('G', '3,').replace('T', '4,')
    seq = re.sub(r'[A-Z]{1}', '-1,', seq)
    return [int(x) for x in seq.split(',')[:-1]]

  df['ref_seq'] = df['ref_seq'].map(seq2int)
  df['alt_seq'] = df['alt_seq'].map(seq2int)
  df['alt_seq2'] = None

  ref_s_list = []
  ref_e_list = []
  alt_e_list = []

  for j in range(len(df)):
      tmp_len = max(len(df['ref_seq'][j]), len(df['alt_seq'][j]))
      pos = df['pos'][j]
      tx_s = df['tx_start'][j]
      tx_e = df['tx_end'][j]
      len_ref = len(df['ref'][j])
      len_alt = len(df['alt'][j])

      if tmp_len < INPUT_LEN:
          alt_seq = df['alt_seq'][j]
          df['ref_seq'][j] += [0 for _ in range(INPUT_LEN-len(df['ref_seq'][j]))]
          df['alt_seq'][j] = alt_seq + [0 for _ in range(INPUT_LEN-len(alt_seq))]
          if len_ref > len_alt:
              df['alt_seq2'][j] = [0 for _ in range(len_ref-len_alt)] + alt_seq + [0 for _ in range(INPUT_LEN-len(alt_seq)-(len_ref-len_alt))]
          elif len_ref < len_alt:
              df['alt_seq2'][j] = alt_seq[len_alt-len_ref:] + [0 for _ in range(INPUT_LEN-len(alt_seq)-(len_ref-len_alt))]
          else:
              df['alt_seq2'][j] = []
          ref_s_list.append(tx_s)
          ref_e_list.append(tx_e)
          alt_e_list.append(tx_e + len_alt - len_ref)
      else:
          strand = df['strand'][j]
          ref_seq = df['ref_seq'][j]
          alt_seq = df['alt_seq'][j]
          max_len = max(len_ref, len_alt)
          dist_1 = (INPUT_LEN-max_len)//2
          dist_2 = INPUT_LEN-dist_1-max_len
          ref_s = pos - dist_1
          ref_e = pos + len_ref + dist_2 - 1
          alt_e = pos + len_alt + dist_2 - 1
          if tx_s > ref_s - abs(len_ref-len_alt) and strand == '-':
              ref_e += tx_s - (ref_s - abs(len_ref-len_alt))
              alt_e += tx_s - (ref_s - abs(len_ref-len_alt))
              ref_s = tx_s +  abs(len_ref-len_alt)
          if tx_e < ref_e + abs(len_ref-len_alt) and strand == '+':
              ref_s += tx_e - (ref_e + abs(len_ref-len_alt))
              alt_e += tx_e - (ref_e + abs(len_ref-len_alt))
              ref_e = tx_e - abs(len_ref - len_alt)
          if tx_e < ref_e and strand == '-':
              ref_s += tx_e - ref_e
              alt_e += tx_e - ref_e
              ref_e = tx_e
          if tx_s > ref_s and strand == '+':
              ref_e += tx_s - ref_s
              alt_e += tx_s - ref_s
              ref_s = tx_s
          ref_s_list.append(ref_s)
          ref_e_list.append(ref_e)
          alt_e_list.append(alt_e)
          if df['strand'][j] == '+':
              df['ref_seq'][j] = ref_seq[ref_s-tx_s:ref_e-tx_s+1]
              df['alt_seq'][j] = alt_seq[ref_s-tx_s:alt_e-tx_s+1]
          else:
              df['ref_seq'][j] = ref_seq[tx_e-ref_e:tx_e-ref_s+1]
              df['alt_seq'][j] = alt_seq[(tx_e+len_alt-len_ref)-alt_e:(tx_e+len_alt-len_ref)-ref_s+1]

          if len_ref < len_alt:
              if df['strand'][j] == '+':
                  df['alt_seq2'][j] = df['alt_seq'][j][len_alt-len_ref:] + alt_seq[alt_e-tx_s+1:][0:len_alt-len_ref]
                  df['ref_seq'][j] += ref_seq[ref_e-tx_s+1:][0:len_alt-len_ref]
              else:
                  df['alt_seq2'][j] = df['alt_seq'][j][len_alt-len_ref:] + alt_seq[(tx_e+len_alt-len_ref)-ref_s+1:][0:len_alt-len_ref]
                  df['ref_seq'][j] += ref_seq[tx_e-ref_s+1:][0:len_alt-len_ref]
          elif len_ref > len_alt:
              if df['strand'][j] == '+':
                  df['alt_seq2'][j] = [0 for _ in range(len_ref-len_alt)] + df['alt_seq'][j]
                  df['alt_seq'][j] += alt_seq[alt_e-tx_s+1:][0:len_ref-len_alt]
              else:
                  df['alt_seq2'][j] = [0 for _ in range(len_ref-len_alt)] + df['alt_seq'][j]
                  df['alt_seq'][j] += alt_seq[(tx_e+len_alt-len_ref)-ref_s+1:][0:len_ref-len_alt]
          else:
              df['alt_seq2'][j] = []

          if len(df['alt_seq2'][j]) != INPUT_LEN and len(df['alt_seq2'][j]) != 0 and len(df['ref_seq'][j]) == len(df['alt_seq2'][j]):
              df['ref_seq'][j] = df['ref_seq'][j] + [0 for _ in range(INPUT_LEN - len(df['ref_seq'][j]))]
              df['alt_seq2'][j] = df['alt_seq2'][j] + [0 for _ in range(INPUT_LEN - len(df['alt_seq2'][j]))]

  df['ref_s'] = ref_s_list
  df['ref_e'] = ref_e_list
  df['alt_e'] = alt_e_list

  return df

def seq_onehot(seq_int):
  seq_raw = tf.constant([seq_int], dtype=tf.int64)  # shape: [1, INPUT_LEN]
  seq_onehot = tf.one_hot(seq_raw, 5)  # shape: [1, INPUT_LEN, 5]
  mask_n = tf.reduce_all(tf.equal(seq_onehot, 0.), axis=-1, keepdims=True)  # shape: [1, INPUT_LEN, 1]
  n_token = tf.tile(0.25 * tf.cast(mask_n, tf.float32), [1,1,4])             # shape: [1, INPUT_LEN, 4]
  seq = seq_onehot[:, :, 1:] + n_token  # shape: [1, INPUT_LEN, 4]

  return seq

def predict_vcf(vcf_file, annotation_file, fasta_dir, model_path, summary_dir, res_file, mask=True):
  if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)
  
  print("Start matching variants to annotation...", flush=True)
  df = match_variant_to_annotation(vcf_file, annotation_file)
  print(f"Number of matched variants: {len(df)}", flush=True)

  print("Start adding sequence...", flush=True)
  df = add_sequence(df, fasta_dir)

  print("Start padding sequence...", flush=True)
  df = pad_sequence(df)
  print("Finished data preparation", flush=True)

  mixed_precision.set_global_policy('mixed_float16')
  model = build_model()
  in_enc_dummy = tf.constant(0., shape=[1, INPUT_LEN, 4])
  model(in_enc_dummy)
  model.load_weights(model_path)

  print(f"Start prediction of {vcf_file}...", flush=True)

  df_res = df[['chrom', 'pos', 'ref', 'alt', 'gene', 'strand']]

  diff_max_list = []
  for i in range(len(df)):
    ref_seq = seq_onehot(df['ref_seq'][i])
    alt_seq = seq_onehot(df['alt_seq'][i])
    if df['alt_seq2'][i] == []:
      alt_seq2 = None
    else:
      alt_seq2 = seq_onehot(df['alt_seq2'][i])

    pred_ref, _, _, _ = model(ref_seq, training=False) # B * (L+16) * 3
    pred_alt, _, _, _ = model(alt_seq, training=False) # B * (L+16) * 3
    if alt_seq2 is not None:
      pred_alt2, _, _, _ = model(alt_seq2, training=False)

    pos = df['pos'][i]
    tx_s = df['tx_start'][i]
    tx_e = df['tx_end'][i]
    len_ref = len(df['ref'][i])
    len_alt = len(df['alt'][i])
    strand = df['strand'][i]
    ref_s = df['ref_s'][i]
    ref_e = df['ref_e'][i]
    alt_e = df['alt_e'][i]
    exon_s = np.array(df['exon_starts'][i][1:]).astype(int)
    exon_e = np.array(df['exon_ends'][i][:-1]).astype(int)
    exon_s = exon_s[(ref_s <= exon_s) & (ref_e >= exon_s)]
    exon_e = exon_e[(ref_s <= exon_e) & (ref_e >= exon_e)]

    mask_ref = tf.reduce_any(tf.not_equal(ref_seq[0], 0.), axis=-1)
    mask_alt = tf.reduce_any(tf.not_equal(alt_seq[0], 0.), axis=-1)
    if alt_seq2 is not None:
      mask_alt2 = tf.reduce_any(tf.not_equal(alt_seq2[0], 0.), axis=-1)

    ref_pred = tf.boolean_mask(pred_ref[0], mask_ref).numpy()
    alt_pred = tf.boolean_mask(pred_alt[0], mask_alt).numpy()
    if alt_seq2 is not None:
      alt2_pred = tf.boolean_mask(pred_alt2[0], mask_alt2).numpy()
    
    ref = ref_pred[:ref_e-ref_s+1,0:2]
    alt = alt_pred[:alt_e-ref_s+1,0:2]

    new_ref_len = len(ref) + max(0, len_alt - len_ref)
    mask_d = np.zeros(new_ref_len, dtype=np.float32)
    mask_a = np.zeros(new_ref_len, dtype=np.float32)

    if len_ref > len_alt:
      if strand == '+':
        alt2 = alt2_pred[:alt_e-ref_s+1, 0:2]
        pre_mut = alt[:pos-ref_s]
        mut = (alt[pos-ref_s:(pos+len_alt)-ref_s] + alt2[pos-ref_s:(pos+len_alt)-ref_s]) / 2
        post_mut = alt2[(pos+len_alt)-ref_s:]
        alt = np.concatenate([pre_mut, mut, np.zeros((len_ref-len_alt, 2)), post_mut], axis=0)

        mask_d[exon_e-ref_s] = 1.
        mask_a[exon_s-ref_s] = 1.
      else:
        alt2 = alt2_pred[:alt_e-ref_s+1, 0:2]
        pre_mut = alt[:alt_e-(pos+len_alt)+1]
        mut = (alt[alt_e-(pos+len_alt)+1:alt_e-pos+1] + alt2[alt_e-(pos+len_alt)+1:alt_e-pos+1]) / 2
        post_mut = alt2[alt_e-pos+1:]
        alt = np.concatenate([pre_mut, mut, np.zeros((len_ref-len_alt, 2)), post_mut], axis=0)

        mask_d[ref_e-exon_s] = 1.
        mask_a[ref_e-exon_e] = 1.
    elif len_alt > len_ref:
      if strand == '+':
        ref = np.concatenate([ref[:(pos+len_ref)-ref_s], np.zeros((len_alt-len_ref, 2)), ref[(pos+len_ref)-ref_s:ref_e-ref_s+1]], axis=0)
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
        mask_d[exon_e-ref_s] = 1.
        mask_a[exon_s-ref_s] = 1.
      else:
        ref = np.concatenate([ref[:ref_e-(pos+len_ref)+1], np.zeros((len_alt-len_ref, 2)), ref[ref_e-(pos+len_ref)+1:ref_e-ref_s+1]], axis=0)
        alt2 = alt2_pred[:alt_e-ref_s+1-(len_alt-len_ref), 0:2]
        pre_mut = alt[:alt_e-(pos+len_alt)+1]
        mut = (alt[alt_e-(pos+len_alt)+1:alt_e-pos+1] +  alt2[alt_e-(pos+len_alt)+1+(len_ref-len_alt):alt_e-pos+1+(len_ref-len_alt)]) / 2
        post_mut = alt2[alt_e-pos+1+len_ref-len_alt:]
        alt = np.concatenate([pre_mut, mut, post_mut], axis=0)

        exon_e = np.where(exon_e > pos+len_ref-1, exon_e+len_alt-len_ref, exon_e)
        exon_s = np.where(exon_s > pos+len_ref-1, exon_s+len_alt-len_ref, exon_s)
        mask_d[alt_e-exon_s] = 1.
        mask_a[alt_e-exon_e] = 1.
    else:
      if strand == '+':
        mask_d[exon_e-ref_s] = 1.
        mask_a[exon_s-ref_s] = 1.
      else:
        mask_d[ref_e-exon_s] = 1.
        mask_a[ref_e-exon_e] = 1.

    diff = alt - ref
    if mask:
      pred_mask = np.max([abs(diff[:,0] * (1. - mask_d)), abs(diff[:,1] * (1. - mask_a))])
      diff_max_list.append(pred_mask)
    else:
      pred_no_mask = np.max(abs(diff))
      diff_max_list.append(pred_no_mask)

  df_res['pred_diff'] = diff_max_list
  df_res.to_csv(os.path.join(summary_dir, res_file), sep="\t", index=False)

  print(f"Results saved to {res_file}", flush=True)
  return


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_file', type=str)
  parser.add_argument('--summary_dir', type=str)
  parser.add_argument('--res_file', type=str)
  parser.add_argument('--gpu', type=int, default=None)
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--annotation', type=str, default='grch37.txt')
  parser.add_argument('--fasta_dir', type=str, default='hg19')
  parser.add_argument('--mask', action='store_true', default=True)
  parser.add_argument('--attention', action='store_true', default=False)
  parser.add_argument('--vcf', action='store_true', default=False)
  parser.add_argument('--model_path', type=str, default=None)

  args = parser.parse_args()

  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

  test_file = args.test_file
  batch_size = args.batch_size
  model_path = args.model_path
  summary_dir = args.summary_dir
  res_file = args.res_file
  annotation_file = args.annotation
  fasta_dir = args.fasta_dir

  if args.vcf:
    predict_vcf(test_file, annotation_file, fasta_dir, model_path, summary_dir, res_file, args.mask)
  elif args.attention:
    predict_attn(test_file, model_path, summary_dir, batch_size, res_file)
  else:
    predict(test_file, model_path, summary_dir, batch_size, res_file)

  return

if __name__ == "__main__":
  main()