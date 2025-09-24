import pandas as pd
import argparse
import os
import tensorflow as tf
import re

input_len = 100000

def seq2int(seq):
    seq = seq.replace('A', '1,').replace('C', '2,').replace('G', '3,').replace('T', '4,')
    seq = re.sub(r'[A-Z]{1}', '-1,', seq)
    return [int(x) for x in seq.split(',')[:-1]]

def make_pad(df):
    df['seq'] = df['seq'].map(seq2int)
    df['da'] = df['da'].map(lambda x: [int(y) for y in x])
    df['ie'] = df['ie'].map(lambda x: [int(y) for y in x])

    index_1 = []
    index_2 = []
    seq = []
    da = []
    ie = []

    for i in range(len(df)):
        tmp_len = len(df['seq'][i])
        if tmp_len < input_len:
            df['seq'][i] += [0 for _ in range(input_len-tmp_len)]
            df['da'][i] += [3 for _ in range(input_len-tmp_len)]
            df['ie'][i] += [-1 for _ in range(input_len-tmp_len)]
        elif tmp_len > input_len:
            for j in range(1, (tmp_len-1)//input_len+1):
                index_1.append(i)
                index_2.append(j)
                if j == (tmp_len-1)//input_len:
                    tmp_start = tmp_len - input_len
                    tmp_end = tmp_len
                else:
                    tmp_start = j * input_len
                    tmp_end = (j+1) * input_len
                tmp_seq = df['seq'][i][tmp_start:tmp_end]
                seq.append(tmp_seq)
                tmp_da = df['da'][i][tmp_start:tmp_end]
                da.append(tmp_da)
                tmp_ie = df['ie'][i][tmp_start:tmp_end]
                ie.append(tmp_ie)
            df['seq'][i] = df['seq'][i][0:input_len]
            df['da'][i] = df['da'][i][0:input_len]
            df['ie'][i] = df['ie'][i][0:input_len]

    df_append = pd.DataFrame({'index_1': index_1, 'index_2': index_2, 'seq': seq, 'da': da, 'ie': ie})
    df['index_1'] = list(df.index)
    df['index_2'] = 0
    df = df[['index_1', 'index_2', 'seq', 'da', 'ie']]
    df_new = df.append(df_append)
    df_new = df_new.sort_values(['index_1', 'index_2'])
    df_new = df_new.reset_index(drop=True)
    return df_new

def convert_to_tf_example(seq, da, ie):
    return tf.train.Example(features=tf.train.Features(feature={
        'seq': tf.train.Feature(int64_list=tf.train.Int64List(value=seq)),
        'da': tf.train.Feature(int64_list=tf.train.Int64List(value=da)),
        'ie': tf.train.Feature(int64_list=tf.train.Int64List(value=ie)),
    }))

def make_tfrecord(df, out_file):
    with tf.io.TFRecordWriter(out_file) as writer:
        for i in range(len(df)):
            seq = df['seq'][i]
            da = df['da'][i]
            ie = df['ie'][i]
            example = convert_to_tf_example(seq, da, ie)
            writer.write(example.SerializeToString())
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--in_file', type=str)
    parser.add_argument('--out_file', type=str)

    args = parser.parse_args()

    data_dir = args.data_dir
    in_file = args.in_file
    out_file = args.out_file

    in_file = os.path.join(data_dir, in_file)
    out_file = os.path.join(data_dir, out_file)
    df = pd.read_csv(in_file, sep='\t')
    df_pad = make_pad(df)
    make_tfrecord(df_pad, out_file)
    
    return

if __name__ == "__main__":
  main()