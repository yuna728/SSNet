import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision
from tensorflow.keras.optimizers import Adam
import datetime
from tqdm import tqdm
import os
import argparse
import json
import pickle

from model import SSNet
from metrics import MaskAccuracy, MaskPrecision, MaskRecall, MaskAccuracy_DA, MaskRecall_Don, MaskRecall_Acc, MaskPrecision_Don, MaskPrecision_Acc
from loss import focal_loss, focal_loss_da

INPUT_LEN = 100000

def parse_record(example):    
    context_features = {
        "seq" : tf.io.FixedLenFeature([input_len], dtype=tf.int64),
        "da" : tf.io.FixedLenFeature([input_len], dtype=tf.int64),
        "ie" : tf.io.FixedLenFeature([input_len], dtype=tf.int64),
    }

    context_parsed = tf.io.parse_example(serialized=example, features=context_features)
    
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

    return {"seq":in_encoder, "da": da, "ie": out_label}

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

def train(train_file, valid_file, checkpoint_dir, summary_dir, batch_size, num_epoch, model_path, opt_path, gamma, alpha1, alpha2, init_lr, start_epoch, start_step, summary_step):
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

  # load dataset
  dataset_train = tf.data.TFRecordDataset(train_file) \
                    .map(parse_record)  \
                    .shuffle(batch_size * 100) \
                    .batch(batch_size, drop_remainder=True).prefetch(2)

  dataset_valid = tf.data.TFRecordDataset(valid_file) \
                    .map(parse_record) \
                    .batch(batch_size, drop_remainder=True).prefetch(2)

  # set mixed precision
  mixed_precision.set_global_policy('mixed_float16')

  # load model architecture
  model = build_model()

  # defin loss function
  loss_fn = focal_loss(gamma, alpha1)
  loss_fn_da = focal_loss_da(gamma, alpha2)

  # define metrics for training
  train_loss = tf.keras.metrics.Mean('loss_train', dtype=tf.float32)
  train_loss_da = tf.keras.metrics.Mean('loss_train_da', dtype=tf.float32)
  train_acu = MaskAccuracy('acu_train')
  train_recall = MaskRecall('recall_train')
  train_precision = MaskPrecision('precision_train')
  train_acu_da = MaskAccuracy_DA('acu_train_da')
  train_recall_d = MaskRecall_Don('recall_train_don')
  train_recall_a = MaskRecall_Acc('recall_train_acc')
  train_precision_d = MaskPrecision_Don('precision_train_don')
  train_precision_a = MaskPrecision_Acc('precision_train_acc')

  # define metrics for validation
  valid_loss = tf.keras.metrics.Mean('loss_valid', dtype=tf.float32)
  valid_loss_da = tf.keras.metrics.Mean('loss_valid_da', dtype=tf.float32)
  valid_acu = MaskAccuracy('acu_valid')
  valid_recall = MaskRecall('recall_valid')
  valid_precision = MaskPrecision('precision_valid')
  valid_acu_da = MaskAccuracy_DA('acu_valid_da')
  valid_recall_d = MaskRecall_Don('recall_valid_don')
  valid_recall_a = MaskRecall_Acc('recall_valid_acc')
  valid_precision_d = MaskPrecision_Don('precision_valid_don')
  valid_precision_a = MaskPrecision_Acc('precision_valid_acc')

  train_ie_metrics = [train_acu, train_recall, train_precision]
  train_da_metrics = [train_acu_da, train_recall_d, train_recall_a, train_precision_d, train_precision_a]
  valid_ie_metrics = [valid_acu, valid_recall, valid_precision, ]
  valid_da_metrics = [valid_acu_da, valid_recall_d, valid_recall_a, valid_precision_d, valid_precision_a]

  # load model if pre-training model is available
  if model_path:
    in_enc_dummy = tf.constant(0., shape=[1, INPUT_LEN, 4])
    model(in_enc_dummy)
    model.load_weights(model_path)

  optimizer = Adam(lr=0., beta_1=0.9, beta_2=0.999, clipnorm=1.0) #0.001
  # load optimizer if pre-training optimizer is available
  if opt_path:
    with open(os.path.join(opt_path), mode="rb") as f:
      opt_weight = pickle.load(f)
    optimizer.build(model.trainable_weights)
    optimizer.set_weights(opt_weight)

  optimizer = mixed_precision.LossScaleOptimizer(inner_optimizer)
  num_warmup_steps = 80000

  @tf.function
  def train_step(batch):
    with tf.GradientTape() as tape:
      pred_da, pred_ie, _, _ = model(batch["seq"], training=True)
      loss = loss_fn(batch["ie"], pred_ie)
      loss_da = loss_fn_da(batch["da"], pred_da)
      scaled_loss = optimizer.get_scaled_loss(loss+loss_da)
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_loss_da(loss_da)
    for metric in train_ie_metrics:
      metric(batch["ie"], pred_ie)
    for metric in train_da_metrics:
      metric(batch["da"], pred_da)

  @tf.function
  def valid_step(batch):
    pred_da, pred_ie, _, _ = model(batch["seq"], training=False)
    loss = loss_fn(batch["ie"], pred_ie)
    loss_da = loss_fn_da(batch["da"], pred_da)

    valid_loss(loss)
    valid_loss_da(loss_da)
    for metric in valid_ie_metrics:
      metric(batch["ie"], pred_ie)
    for metric in valid_da_metrics:
      metric(batch["da"], pred_da)

  def write_train_summary(writer, loss_list, metrics, step, epoch):
    with writer.as_default():
      for loss in loss_list:
        tf.summary.scalar(loss.name, loss.result(), step=step)
      for metric in metrics:
        tf.summary.scalar(metric.name, metric.result(), step=step)
    print("Epochs: %02d, Steps: %06d, " % (epoch, step), flush=True, end="")
    for loss in loss_list:
      print("%s: %.4e" % (loss.name, float(loss.result())), end=", ")
    for metric in metrics:
      print("%s: %.4f" % (metric.name, float(metric.result())), end=", ")
    print("")

  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  train_log_dir = os.path.join(summary_dir,'history_'+current_time)
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)

  global_step = start_step

  lr = init_lr * (1.0- tf.cast(global_step, tf.float32)/800000.)
  is_warmup = tf.cast(global_step < num_warmup_steps, tf.float32)
  lr = (1.0 - is_warmup) * lr + is_warmup * (init_lr * tf.cast(global_step, tf.float32)/tf.cast(num_warmup_steps, tf.float32))
  optimizer.learning_rate = lr

  for epoch in range(start_epoch, start_epoch + num_epoch):
    print("Start of epoch %d" % (epoch+1), flush=True)

    train_loss.reset_states()
    train_loss_da.reset_states()
    for metric in train_ie_metrics + train_da_metrics:
      metric.reset_states()

    valid_loss.reset_states()
    valid_loss_da.reset_states()
    for metric in valid_ie_metrics + valid_da_metrics:
      metric.reset_states()

    for step, batch_train in enumerate(tqdm(dataset_train, mininterval=5, desc="[train]")):
      train_step(batch_train)

      global_step += 1

      if global_step % summary_step == 0:
        write_train_summary(train_summary_writer, [train_loss, train_loss_da], train_ie_metrics+train_da_metrics, step=global_step, epoch=epoch+1)

      # update learning rate
      lr = init_lr * (1.0- tf.cast(global_step, tf.float32)/800000.)
      is_warmup = tf.cast(global_step < num_warmup_steps, tf.float32)
      lr = (1.0 - is_warmup) * lr + is_warmup * (init_lr * tf.cast(global_step, tf.float32)/tf.cast(num_warmup_steps, tf.float32))
      optimizer.learning_rate = lr

    print("")

    # validation
    for step, batch_valid in enumerate(tqdm(dataset_valid, mininterval=5, desc="[valid]")):
      valid_step(batch_valid)

    print("-----Validation-----")
    print("Epochs: %02d, " % (epoch+1), end="")
    for loss in [valid_loss, valid_loss_da]:
      print("%s: %.4e" % (loss.name, float(loss.result())), end=", ")
    for metric in valid_ie_metrics + valid_da_metrics:
      print("%s: %.4f" % (metric.name, float(metric.result())), end=", ")
    print("")
    print("", flush=True)

    # save model
    model.save_weights(os.path.join(checkpoint_dir,'model_{epoch:02d}_{step:d}.h5'.format(epoch=epoch+1, step=global_step)))
    print("-----Save Model-----")
    print("Saved Model : model_%02d_%d.h5" % (epoch+1, global_step))
    print("", flush=True)

    with open(os.path.join(checkpoint_dir,'opt_{epoch:02d}_{step:d}.pkl'.format(epoch=epoch+1, step=global_step)), mode="wb") as f:
      pickle.dump(optimizer.variables(), f)
    print("-----Save Optimizer-----")
    print("Saved Optimizer : opt_%02d_%d.pkl" % (epoch+1, global_step))
    print("")
    print("", flush=True)
    
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_file', type=str)
  parser.add_argument('--valid_file', type=str)
  parser.add_argument('--checkpoint_dir', type=str)
  parser.add_argument('--summary_dir', type=str)
  parser.add_argument('--gpu', type=int, default=None)
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--summary_step', type=int, default=10)
  parser.add_argument('--num_epoch', type=int, default=20)
  parser.add_argument('--model_path', type=str, default=None)
  parser.add_argument('--opt_path', type=str, default=None)
  parser.add_argument('--start_epoch', type=int, default=0)
  parser.add_argument('--start_step', type=int, default=0)
  parser.add_argument('--gamma', type=float, default=2.0)
  parser.add_argument('--alpha1', nargs="*", type=float)
  parser.add_argument('--alpha2', nargs="*", type=float)
  parser.add_argument('--lr', type=float, default=0.00025)

  args = parser.parse_args()

  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

  train_file = args.train_file
  valid_file = args.valid_file
  batch_size = args.batch_size
  summary_step = args.summary_step
  num_epoch = args.num_epoch
  model_path = args.model_path
  opt_path = args.opt_path
  gamma = args.gamma
  alpha1 = args.alpha1
  alpha2 = args.alpha2
  lr = args.lr

  start_epoch = args.start_epoch
  start_step = args.start_step
  
  checkpoint_dir = args.checkpoint_dir
  summary_dir = args.summary_dir
  train(train_file, valid_file, checkpoint_dir, summary_dir, batch_size, num_epoch, model_path, opt_path, gamma, alpha1, alpha2, lr, start_epoch, start_step, summary_step)

  return

if __name__ == "__main__":
  main()