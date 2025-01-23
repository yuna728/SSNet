import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Conv1D, BatchNormalization, ReLU, LayerNormalization, Reshape

INPUT_LEN = 100000

class ResidualBlock(Layer):
  def __init__(self, out_dim, kernel_size):
    super(ResidualBlock, self).__init__()
    self.out_dim = out_dim
    self.kernel_size = kernel_size

    self.conv1 = Conv1D(out_dim, kernel_size, padding="same")
    self.bn1 = BatchNormalization()
    self.relu1 = ReLU()
    self.conv2 = Conv1D(out_dim, kernel_size, padding="same")
    self.bn2 = BatchNormalization()
    self.relu2 = ReLU()

  def call(self, inputs, training=None):
    conv1_out = self.conv1(inputs)
    bn1_out = self.bn1(conv1_out)
    relu1_out = self.relu1(bn1_out)
    conv2_out = self.conv2(relu1_out)
    bn2_out = self.bn2(conv2_out)
    relu2_out = self.relu2(bn2_out)
    return inputs + relu2_out

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float16)

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float16)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += mask * (tf.float16.min / 10.0)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(tf.cast(scaled_attention_logits, tf.float32), axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(tf.cast(attention_weights, tf.float16), v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class GlobalAttention(Layer):
  def __init__(self, d_model, num_heads):
    super(GlobalAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = Dense(d_model)
    self.wk = Dense(d_model)
    self.wv = Dense(d_model)

    self.dense = Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

  def get_config(self):
    config = {'d_model': self.d_model, 'num_heads': self.num_heads}
    base_config = super(GlobalAttention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

def relative_shift(x):
  to_pad = tf.zeros_like(x[..., :1]) # (batch_size, num_areas, num_heads, l_q, 1)
  x = tf.concat([to_pad, x], -1) # (batch_size, num_areas, num_heads, l_q, 2*l_q)
  _, num_areas, num_heads, t1, t2 = x.shape
  x = tf.reshape(x, [-1, num_areas, num_heads, t2, t1]) # (batch_size, num_areas, num_heads, 2*l_q, l_q)
  x = tf.slice(x, [0, 0, 0, 1, 0], [-1, -1, -1, -1, -1]) # (batch_size, num_areas, num_heads, 2*l_q-1, l_q)
  x = tf.reshape(x, [-1, num_areas, num_heads, t1, t2 - 1]) # (batch_size, num_areas, num_hedas, l_q, 2*l_q-1)
  x = tf.slice(x, [0, 0, 0, 0, 0], [-1, -1, -1, -1, t2 // 2]) # (batch_size, num_areas, num_heads, l_q, l_q)
  return x

class CompressAttention(Layer):
  def __init__(self, d_model, num_areas, num_heads, rate=0.1):
    super(CompressAttention, self).__init__()
    self.num_areas = num_areas
    self.num_heads = num_heads
    self.d_model = d_model

    assert INPUT_LEN % self.num_areas == 0
    assert d_model % self.num_heads == 0

    self.max_len = 2 * (1 + INPUT_LEN // self.num_areas) -1
    self.depth = d_model // self.num_heads

    self.wq = Dense(d_model)
    self.wk = Dense(d_model)
    self.wv = Dense(d_model)

    self.E = self.add_weight(name='Erel', shape=(self.max_len, self.depth), initializer='random_normal', trainable=True)
    self.dropout = Dropout(rate)

  def split_areas(self, x, batch_size):
    x = tf.reshape(x, (batch_size, self.num_areas, -1, self.d_model))
    return x

  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, self.num_areas, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 1, 3, 2, 4])

  def get_left_embed(self, len_q):
    start_point = tf.math.maximum(0, self.max_len - (2*len_q-1))
    e = self.E[start_point:,:]
    return e

  def call(self, inp, mask, training):
    batch_size = tf.shape(inp)[0]
    seq_len = tf.shape(inp)[1]

    inp_split = self.split_areas(inp, batch_size) # (batch_size, num_areas, seq_len/num_areas, d_model)

    q = self.wq(inp_split)  # (batch_size, num_areas, seq_len/num_areas, d_model)
    k = self.wk(inp_split)  # (batch_size, num_areas, seq_len/num_areas, d_model)
    v = self.wv(inp_split)  # (batch_size, num_areas, seq_len/num_areas, d_model)

    q = self.split_heads(q, batch_size) # (batch_size, num_areas, num_heads, seq_len/num_areas, depth)
    k = self.split_heads(k, batch_size)
    v = self.split_heads(v, batch_size)

    len_q = tf.shape(q)[3] # l_q = seq_len/num_areas

    # relative positional encoding
    E = self.get_left_embed(len_q) # (2*l_q-1, depth)
    E = self.dropout(E, training=training)
    QE = tf.einsum('bahld,md->bahlm', q, E) # (batch_size, num_areas, num_heads, l_q, 2*l_q-1)
    Srel = relative_shift(QE) # (batch_size, num_areas, num_heads, l_q, l_q)

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_areas, num_heads, l_q, l_q)

    # scaling
    logits = matmul_qk + Srel
    scaled_attention_logits = logits / tf.math.sqrt(tf.cast(self.d_model, tf.float16))
    #scaled_attention_logits = matmul_qk / tf.math.sqrt(tf.cast(self.d_model, tf.float32))

    # add the mask to the scaled tensor.
    if mask is not None:
      scaled_attention_logits += mask * (tf.float16.min / 10.0)

    # softmax 
    attention_weights = tf.cast(tf.nn.softmax(tf.cast(scaled_attention_logits, tf.float32), axis=-1), tf.float16)  # (batch_size, num_areas, num_heads, l_q, l_q)

    scaled_attention = tf.matmul(attention_weights, v)  # (batch_size, num_areas, num_heads, l_q, depth)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 1, 3, 2, 4])  # (batch_size, num_areas, l_q, num_heads, depth)
    concat_attention = tf.reshape(scaled_attention, (batch_size, self.num_areas, -1, self.d_model))  # (batch_size, num_areas, l_q, d_model)

    return output, attention_weights

  def get_config(self):
    config = {'d_model': self.d_model, 'num_areas': self.num_areas}
    base_config = super(CompressAttention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class FFN(Layer):
  def __init__(self, d_model, dff):
    super(FFN, self).__init__()
    self.dense1 = Dense(dff, activation="relu")
    self.dense2 = Dense(d_model)
  def call(self, inputs):
    dense1_out = self.dense1(inputs)
    dense2_out = self.dense2(dense1_out)
    return dense2_out
  def get_config(self):
    config = {'d_model': self.d_model, 'dff': self.dff}
    base_config = super(FFN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class CompressLayer(Layer):
  def __init__(self, d_model, num_areas, num_heads, dff, rate=0.1):
    super(CompressLayer, self).__init__()

    self.d_model = d_model
    self.num_areas = num_areas
    self.num_heads = num_heads
    self.dff = dff

    self.ca = CompressAttention(d_model, d_out, num_areas, num_heads)
    self.ffn = FFN(d_out, dff)

    self.layernorm1 = LayerNormalization(epsilon=1e-6, dtype=tf.float32)
    self.layernorm2 = LayerNormalization(epsilon=1e-6, dtype=tf.float32)

    self.dropout1 = Dropout(rate)
    self.dropout2 = Dropout(rate)
    self.dropout3 = Dropout(rate)

  def call(self, x, training, mask):
    attn_output, attn_weights = self.ca(x, mask, training=training)  # (batch_size, num_areas, d_out)
    attn_output = self.dropout2(attn_output, training=training)
    out1 = self.layernorm1(simp_comp + attn_output)  # (batch_size, num_areas, d_out)
    out1 = tf.cast(out1, tf.float16)

    ffn_output = self.ffn(out1)  # (batch_size, num_areas, d_out)
    ffn_output = self.dropout3(ffn_output, training=training)
    out2 = self.layernorm2(simp_comp + out1 + ffn_output)  # (batch_size, num_areas, d_out)
    out2 = tf.cast(out2, tf.float16)

    return out2, attn_weights

  def get_config(self):
    config = {'d_model': self.d_model, 'd_out': self.d_out, 'num_areas': self.num_areas, 'num_heads': self.num_heads, 'dff': self.dff}
    base_config = super(CompressLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class EncoderLayer(Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = FFN(d_model, dff)

    self.layernorm1 = LayerNormalization(epsilon=1e-6, dtype=tf.float32)
    self.layernorm2 = LayerNormalization(epsilon=1e-6, dtype=tf.float32)

    self.dropout1 = Dropout(rate)
    self.dropout2 = Dropout(rate)

  def call(self, x, training, mask):
    attn_output, attn_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    out1 = tf.cast(out1, tf.float16)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(x + out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    out2 = tf.cast(out2, tf.float16)

    return out2, attn_weights

  def get_config(self):
    config = {'d_model': self.d_model, 'num_heads': self.num_heads, 'dff': self.dff}
    base_config = super(EncoderLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

