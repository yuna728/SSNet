import tensorflow as tf
import tensorflow.keras.backend as K

def focal_loss(gamma, alpha):
  def fixed_focal_loss(y_true, y_pred):
    y_true = tf.squeeze(y_true)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, 3])
    
    p_t = tf.gather(y_pred, y_true, axis=-1, batch_dims=1)
    focal_factor = tf.pow(1.0-p_t, gamma)

    logits = tf.math.log(tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon()))
    xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    focal_ce = focal_factor * xent_loss

    class_weight = tf.constant(alpha)
    class_weight = tf.gather(class_weight, y_true, axis=0, batch_dims=1)
    focal_ce = class_weight * focal_ce
    return K.mean(focal_ce, axis=-1)
  return fixed_focal_loss

def focal_loss_da(gamma, alpha):
  def fixed_focal_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, 4])
    
    p_t = tf.gather(y_pred, y_true, axis=-1, batch_dims=1)
    focal_factor = tf.pow(1.0-p_t, gamma)

    logits = tf.math.log(tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon()))
    xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    focal_ce = focal_factor * xent_loss

    class_weight = tf.constant(alpha)
    class_weight = tf.gather(class_weight, y_true, axis=0, batch_dims=1)
    focal_ce = class_weight * focal_ce
    return K.mean(focal_ce, axis=-1)
  return fixed_focal_loss
