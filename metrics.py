import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric

class MaskAccuracy(Metric):
  def __init__(self, name='mask_acu', **kwargs):
    super(MaskAccuracy, self).__init__(name=name, **kwargs)
    self.matched = tf.Variable(0.)
    self.unmatched = tf.Variable(0.)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, 3])

    mask = tf.squeeze(tf.not_equal(y_true, 0))

    y_true_argmax = tf.boolean_mask(y_true, mask)
    y_pred_argmax = tf.argmax(y_pred, axis=-1)
    y_pred_argmax = tf.boolean_mask(y_pred_argmax, mask)

    y_matched = tf.reduce_sum(tf.cast(tf.equal(y_true_argmax, y_pred_argmax), tf.float32))
    y_unmatched = tf.reduce_sum(tf.cast(tf.not_equal(y_true_argmax, y_pred_argmax), tf.float32))
    
    self.matched.assign_add(y_matched)
    self.unmatched.assign_add(y_unmatched)
                       
  def result(self):
    return self.matched / (self.matched + self.unmatched)

  def reset_states(self):
    self.matched.assign(0.)
    self.unmatched.assign(0.)

class MaskRecall(Metric):
  def __init__(self, name='mask_recall', **kwargs):
    super(MaskRecall, self).__init__(name=name, **kwargs)
    self.true_positives = tf.Variable(0.)
    self.total_positives = tf.Variable(0.)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, 3])

    mask = tf.squeeze(tf.not_equal(y_true, 0))

    y_true_argmax = tf.boolean_mask(y_true, mask)
    y_pred_argmax = tf.argmax(y_pred, axis=-1)
    y_pred_argmax = tf.boolean_mask(y_pred_argmax, mask)

    y_true_argmax = y_true_argmax-1
    y_pred_argmax = tf.maximum(tf.constant(0, dtype=tf.int64), y_pred_argmax-1)

    true_p = tf.cast(tf.reduce_sum(y_true_argmax * y_pred_argmax), tf.float32)
    total_p = tf.cast(tf.reduce_sum(y_true_argmax), tf.float32)
 
    self.true_positives.assign_add(true_p)
    self.total_positives.assign_add(total_p)
                       
  def result(self):
    return self.true_positives / (self.total_positives + K.epsilon())

  def reset_states(self):
    self.true_positives.assign(0.)
    self.total_positives.assign(0.)

class MaskPrecision(Metric):
  def __init__(self, name='mask_precision', **kwargs):
    super(MaskPrecision, self).__init__(name=name, **kwargs)
    self.true_positives = tf.Variable(0.)
    self.total_positives = tf.Variable(0.)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, 3])

    mask = tf.squeeze(tf.not_equal(y_true, 0))

    y_true_argmax = tf.boolean_mask(y_true, mask)
    y_pred_argmax = tf.argmax(y_pred, axis=-1)
    y_pred_argmax = tf.boolean_mask(y_pred_argmax, mask)

    y_true_argmax = y_true_argmax-1
    y_pred_argmax = tf.maximum(tf.constant(0, dtype=tf.int64), y_pred_argmax-1)

    true_p = tf.cast(tf.reduce_sum(y_true_argmax * y_pred_argmax), tf.float32)
    total_p = tf.cast(tf.reduce_sum(y_pred_argmax), tf.float32)
 
    self.true_positives.assign_add(true_p)
    self.total_positives.assign_add(total_p)
                       
  def result(self):
    return self.true_positives / (self.total_positives + K.epsilon())

  def reset_states(self):
    self.true_positives.assign(0.)
    self.total_positives.assign(0.)

class MaskAccuracy_DA(Metric):
  def __init__(self, name='mask_acu', **kwargs):
    super(MaskAccuracy_DA, self).__init__(name=name, **kwargs)
    self.matched = tf.Variable(0.)
    self.unmatched = tf.Variable(0.)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(y_true, [-1]) # B * L
    y_pred = tf.reshape(y_pred, [-1, 4]) # B * L * 4

    mask = tf.squeeze(tf.not_equal(y_true, 3)) # Donor: 0, Acceptor: 1, None: 2, Pad: 3
    y_true_mask = tf.boolean_mask(y_true, mask)
    y_pred_argmax = tf.argmax(y_pred, axis=-1)
    y_pred_mask = tf.boolean_mask(y_pred_argmax, mask)

    y_matched = tf.reduce_sum(tf.cast(tf.equal(y_true_mask, y_pred_mask), tf.float32))
    y_unmatched = tf.reduce_sum(tf.cast(tf.not_equal(y_true_mask, y_pred_mask), tf.float32))
    
    self.matched.assign_add(y_matched)
    self.unmatched.assign_add(y_unmatched)
                       
  def result(self):
    return self.matched / (self.matched + self.unmatched)

  def reset_states(self):
    self.matched.assign(0.)
    self.unmatched.assign(0.)

class MaskRecall_Don(Metric):
  def __init__(self, name='mask_recall', **kwargs):
    super(MaskRecall_Don, self).__init__(name=name, **kwargs)
    self.true_positives = tf.Variable(0.)
    self.total_positives = tf.Variable(0.)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, 4])

    mask = tf.squeeze(tf.not_equal(y_true, 3))
    y_true_mask = tf.boolean_mask(y_true, mask) + 1 # Donor: 1, Acceptor: 2, None: 3
    y_pred_argmax = tf.argmax(y_pred, axis=-1)
    y_pred_mask = tf.boolean_mask(y_pred_argmax, mask) + 1 # Donor: 1, Acceptor: 2, None: 3, Pad: 4

    y_true = tf.where(y_true_mask >= 2, tf.constant(0, dtype=tf.int64), y_true_mask) # Donor: 1, Other: 0
    y_pred = tf.where(y_pred_mask >= 2, tf.constant(0, dtype=tf.int64), y_pred_mask)

    true_p = tf.cast(tf.reduce_sum(y_true * y_pred), tf.float32)
    total_p = tf.cast(tf.reduce_sum(y_true), tf.float32)
 
    self.true_positives.assign_add(true_p)
    self.total_positives.assign_add(total_p)
                       
  def result(self):
    return self.true_positives / (self.total_positives + K.epsilon())

  def reset_states(self):
    self.true_positives.assign(0.)
    self.total_positives.assign(0.)

class MaskRecall_Acc(Metric):
  def __init__(self, name='mask_recall', **kwargs):
    super(MaskRecall_Acc, self).__init__(name=name, **kwargs)
    self.true_positives = tf.Variable(0.)
    self.total_positives = tf.Variable(0.)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, 4])

    mask = tf.squeeze(tf.not_equal(y_true, 3))
    y_true_mask = tf.boolean_mask(y_true, mask) # Donor: 0, Acceptor: 1, None: 2
    y_pred_argmax = tf.argmax(y_pred, axis=-1)
    y_pred_mask = tf.boolean_mask(y_pred_argmax, mask) # Donor: 0, Acceptor: 1, None: 2, Pad: 3

    y_true = tf.where(y_true_mask == 2, tf.constant(0, dtype=tf.int64), y_true_mask) # Acceptor: 1, Other: 0
    y_pred = tf.where(y_pred_mask >= 2, tf.constant(0, dtype=tf.int64), y_pred_mask) # Acceptor: 1, Other: 0

    true_p = tf.cast(tf.reduce_sum(y_true * y_pred), tf.float32)
    total_p = tf.cast(tf.reduce_sum(y_true), tf.float32)
 
    self.true_positives.assign_add(true_p)
    self.total_positives.assign_add(total_p)
                       
  def result(self):
    return self.true_positives / (self.total_positives + K.epsilon())

  def reset_states(self):
    self.true_positives.assign(0.)
    self.total_positives.assign(0.)


class MaskPrecision_Don(Metric):
  def __init__(self, name='mask_precision', **kwargs):
    super(MaskPrecision_Don, self).__init__(name=name, **kwargs)
    self.true_positives = tf.Variable(0.)
    self.total_positives = tf.Variable(0.)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, 4])

    mask = tf.squeeze(tf.not_equal(y_true, 3))
    y_true_mask = tf.boolean_mask(y_true, mask) + 1 # Donor: 1, Acceptor: 2, None: 3
    y_pred_argmax = tf.argmax(y_pred, axis=-1)
    y_pred_mask = tf.boolean_mask(y_pred_argmax, mask) + 1 # Donor: 1, Acceptor: 2, None: 3, Pad: 4

    y_true = tf.where(y_true_mask >= 2, tf.constant(0, dtype=tf.int64), y_true_mask) # Donor: 1, Other: 0
    y_pred = tf.where(y_pred_mask >= 2, tf.constant(0, dtype=tf.int64), y_pred_mask)

    true_p = tf.cast(tf.reduce_sum(y_true * y_pred), tf.float32)
    total_p = tf.cast(tf.reduce_sum(y_pred), tf.float32)
 
    self.true_positives.assign_add(true_p)
    self.total_positives.assign_add(total_p)
                       
  def result(self):
    return self.true_positives / (self.total_positives + K.epsilon())

  def reset_states(self):
    self.true_positives.assign(0.)
    self.total_positives.assign(0.)

class MaskPrecision_Acc(Metric):
  def __init__(self, name='mask_precision', **kwargs):
    super(MaskPrecision_Acc, self).__init__(name=name, **kwargs)
    self.true_positives = tf.Variable(0.)
    self.total_positives = tf.Variable(0.)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, 4])

    mask = tf.squeeze(tf.not_equal(y_true, 3))
    y_true_mask = tf.boolean_mask(y_true, mask) # Donor: 0, Acceptor: 1, None: 2
    y_pred_argmax = tf.argmax(y_pred, axis=-1)
    y_pred_mask = tf.boolean_mask(y_pred_argmax, mask) # Donor: 0, Acceptor: 1, None: 2, Pad: 3

    y_true = tf.where(y_true_mask == 2, tf.constant(0, dtype=tf.int64), y_true_mask) # Acceptor: 1, Other: 0
    y_pred = tf.where(y_pred_mask >= 2, tf.constant(0, dtype=tf.int64), y_pred_mask) # Acceptor: 1, Other: 0

    true_p = tf.cast(tf.reduce_sum(y_true * y_pred), tf.float32)
    total_p = tf.cast(tf.reduce_sum(y_pred), tf.float32)
 
    self.true_positives.assign_add(true_p)
    self.total_positives.assign_add(total_p)
                       
  def result(self):
    return self.true_positives / (self.total_positives + K.epsilon())

  def reset_states(self):
    self.true_positives.assign(0.)
    self.total_positives.assign(0.)
