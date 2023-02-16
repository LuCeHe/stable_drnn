import tensorflow as tf
import numpy as np

from alif_sg.neural_models.transformer_model import EncoderLayer, DecoderLayer
from alif_sg.neural_models.transformer_model import build_model

SEQ_MAX_LEN_SOURCE = 100
SEQ_MAX_LEN_TARGET = 101
BPE_VOCAB_SIZE = 32000
encoder_count = 6
decoder_count=encoder_count
batch_size = 2
attention_head_count = 2
d_model = 12
d_point_wise_ff = 2 * d_model
dropout_prob = .2
activation = 'swish'
comments = ''
layer_index = 0

enc = EncoderLayer(
    attention_head_count,
    d_model,
    d_point_wise_ff,
    dropout_prob,
    activation=activation,
    comments=comments,
    layer_index=layer_index
)

dec = DecoderLayer(
    attention_head_count,
    d_model,
    d_point_wise_ff,
    dropout_prob,
    activation=activation,
    comments=comments,
    layer_index=layer_index
)

# here (16, 100, 512) (16, 1, 1, 100)

rep_shape = (batch_size, 3, d_model)
mask_shape = [rep_shape[0]] + [1, 1] + [rep_shape[1]]

input_layer = tf.keras.layers.Input(rep_shape[1:])
mask_layer = tf.keras.layers.Input(mask_shape[1:])

print(input_layer.shape)
print(mask_layer.shape)

output = enc([input_layer, mask_layer])
enc_model = tf.keras.Model(inputs=[input_layer, mask_layer], outputs=output)

# get weights
enc_weights = enc_model.get_weights()

# turn weights to ones
enc_weights = [np.ones_like(w) for w in enc_weights]

# set weights
enc_model.set_weights(enc_weights)

print(output.shape)

decoder_tensor = tf.keras.layers.Input(rep_shape[1:])
encoder_tensor = tf.keras.layers.Input(rep_shape[1:])
target_padding_mask = tf.keras.layers.Input(mask_shape[1:])
output = dec([decoder_tensor, encoder_tensor, target_padding_mask])
model = tf.keras.Model(inputs=[decoder_tensor, encoder_tensor, target_padding_mask], outputs=output)

# get weights
weights = model.get_weights()

# turn weights to ones
# weights = [np.ones_like(w) for w in weights]

# turn weights to binomial noise
weights = [np.random.binomial(1, .5, w.shape) for w in weights]

# set weights
model.set_weights(weights)

weights = model.get_weights()
print(weights[0])

transformer = build_model(
    inputs_timesteps=SEQ_MAX_LEN_SOURCE,
    target_timesteps=SEQ_MAX_LEN_TARGET,
    inputs_vocab_size=BPE_VOCAB_SIZE,
    target_vocab_size=BPE_VOCAB_SIZE,
    encoder_count=encoder_count,
    decoder_count=decoder_count,
    attention_head_count=attention_head_count,
    d_model=d_model,
    d_point_wise_ff=d_point_wise_ff,
    dropout_prob=dropout_prob,
    batch_size=batch_size,
    activation=activation,
    comments=comments,
)

#print names of weights
print([w.name for w in transformer.weights])
print(len(transformer.weights))

# same for the encoder model
print([w.name for w in enc.weights])
print(len(enc.weights))

for i, wt in enumerate(transformer.weights):
    print(i, wt.name)