import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, LayerNormalization, Reshape, Softmax

from layer import ResidualBlock, CompressLayer, EncoderLayer, positional_encoding

class SSNet(Model):
  def __init__(self, conv_dim, conv_kernel, num_c_layers, num_t_layers, d_comp, d_model, num_areas, num_heads_comp, num_heads, dff_comp, dff, pe_input, rate=0.1):
    super(SSNet, self).__init__()

    self.pe_input = pe_input

    self.d_comp = d_comp
    self.d_model = d_model
    self.num_areas = num_areas
    self.num_c_layers = num_c_layers
    self.num_t_layers = num_t_layers
    self.num_c_blocks = len(conv_kernel)

    ### Conv blocks ###
    self.conv1 = Conv1D(conv_dim, 1, padding='same')
    self.conv2 = Conv1D(conv_dim, 1, padding='same')
    self.conv3 = Conv1D(conv_dim, 1, padding='same')
    self.conv_blocks = [ResidualBlock(conv_dim, k) for k in conv_kernel]

    ### Embedding & Pos Encoding 1 ###
    self.embedding1 = Dense(d_comp)
    self.pos_encoding1 = positional_encoding(pe_input, d_comp)
    self.layernorm1 = LayerNormalization(epsilon=1e-6, dtype=tf.float32)
    self.dropout1 = Dropout(0.3)

    ## reshape for compress layer ##
    self.reshape_in_comp = Reshape((num_areas, -1, d_comp))

    ### Compress Length by Attention ###
    self.comp_layers = [CompressLayer(d_comp, num_areas, num_heads_comp, dff_comp, rate)
                        for _ in range(num_c_layers)]

    ### Encoder ###
    self.reshape_out_comp = Reshape((num_areas, -1))
    self.embedding2 = Dense(d_model)
    self.pos_encoding2 = positional_encoding(num_areas, d_model)
    self.layernorm2 = LayerNormalization(epsilon=1e-6, dtype=tf.float32)
    self.dropout2 = Dropout(0.3)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_t_layers)]


    ### Final Layer ###
    self.fc1 = Dense(640)
    self.reshape1 = Reshape((-1, 4))

    self.fc2 = Dense(480)
    self.reshape2 = Reshape((-1, 3))
    self.softmax = Softmax(dtype=tf.float32)

  def call(self, inputs, training=None):
    batch = tf.shape(inputs)[0]
    const_0 = tf.constant(0., shape=[1,1,1], dtype=tf.float16)
    const_1 = tf.constant(1., shape=[1,1,1], dtype=tf.float16)

    mask = tf.tile(const_0, [batch, tf.shape(inputs)[1], 1])
    inp = tf.concat([inputs, mask], axis=-1) # B * L * 5

    inp_mask = tf.reduce_all(tf.equal(inp, 0.), axis=-1) # inp_mask = [B, L]
    comp_mask, enc_mask = self.create_masks(inp_mask, batch)

    ### Conv Tower ###
    x = inp
    x_conv1 = self.conv1(x, training=training)
    x = x_conv1
    for i in range(self.num_c_blocks):
      x = self.conv_blocks[i](x, training=training)
    x_conv2 = self.conv2(x_conv1, training=training)
    x_conv3 = self.conv3(x, training=training)
    x = x_conv2 + x_conv3

    ### adding embedding and position encoding. ###
    inp_len1 = tf.shape(inp)[1]
    x = self.embedding1(x) # B * L * d_comp
    x *= tf.math.sqrt(tf.cast(self.d_comp, tf.float16))
    x += self.pos_encoding1[:, :inp_len1, :]
    x = self.layernorm1(x)
    x = self.dropout1(x, training=training)  # (batch_size, len, d_comp)

    ## reshape for compress layer ##
    x = self.reshape_in_comp(x) # (batch_size, num_areas, len//num_areas, d_comp)

    ### Compress Length by Attention ###
    for i in range(self.num_c_layers):
      x, local_attn = self.comp_layers[i](x, training, comp_mask) # (batch_size, num_areas, len//num_areas, d_comp)
      if i == 0:
        local_attn_ave = local_attn
      else:
        local_attn_ave = (i * local_attn_ave + local_attn) / (i + 1)

    ### adding embedding and position encoding. ###
    inp_len2 = tf.shape(x)[1] # num_areas
    x  = self.reshape_out_comp(x)   # (batch_size, num_areas, len//num_areas * d_comp)
    x = self.embedding2(x) # (batch_size, num_areas, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float16)) # (batch_size, num_areas, d_model)
    x += self.pos_encoding2[:, :inp_len2, :] # (batch_size, num_areas, d_model)
    x = self.layernorm2(x)
    x = self.dropout2(x, training=training)

    ### Encoder ###
    global_attn_list = []
    for i in range(self.num_t_layers):
      x, global_attn = self.enc_layers[i](x, training, enc_mask)
      global_attn_list.append(global_attn)

    out_encoder = x

    ### Final Layer ###
    out1 = self.fc1(out_encoder) # B * 625 * 640
    out1 = self.reshape1(out1) # B * 100000 * 4

    out2 = self.fc2(out_encoder) # B * 625 * 480
    out2 = self.reshape2(out2) # B * 100000 * 3

    ### end ###

    output_da = self.softmax(out1)
    output_ie = self.softmax(out2)

    return output_da, output_ie, local_attn_ave, global_attn_list

  def create_masks(self, inp, batch):
    mask = tf.reshape(inp, [batch, self.num_areas, -1]) # (batch_size, num_areas, seq_len/num_areas)
    comp_mask = mask[:, :, tf.newaxis, tf.newaxis, :] # (batch_size, num_areas, 1, 1, seq_len/num_areas)

    enc_mask = tf.reduce_all(mask, -1) # (batch_size, num_areas)
    enc_mask = enc_mask[:, tf.newaxis, tf.newaxis, :]

    return tf.cast(comp_mask, tf.float16), tf.cast(enc_mask, tf.float16)
