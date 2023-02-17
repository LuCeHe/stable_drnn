import tensorflow as tf
import numpy as np

from alif_sg.neural_models.recLSC import get_norms
from alif_sg.neural_models.transformer_model import EncoderLayer, DecoderLayer
from alif_sg.neural_models.transformer_model import build_model
from tensorflow_addons.optimizers import AdamW

SEQ_MAX_LEN_SOURCE = 2
SEQ_MAX_LEN_TARGET = 3
BPE_VOCAB_SIZE = 2
encoder_count = 2
decoder_count = encoder_count
batch_size = 2
attention_head_count = 2
d_model = 4
d_point_wise_ff = 2 * d_model
dropout_prob = .2
activation = 'swish'
comments = 'meanaxis_supsubnpsd'
layer_index = 0
epochs = 5
lr = 1e-3

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

rep_shape = (batch_size, SEQ_MAX_LEN_SOURCE, d_model)
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
dec_model = tf.keras.Model(inputs=[decoder_tensor, encoder_tensor, target_padding_mask], outputs=output)

# get weights
weights = dec_model.get_weights()

# turn weights to ones
# weights = [np.ones_like(w) for w in weights]

# turn weights to binomial noise
weights = [np.random.binomial(1, .5, w.shape) for w in weights]

# set weights
dec_model.set_weights(weights)

weights = dec_model.get_weights()
# print(weights[0])

coders = {'encoder': enc_model, 'decoder': dec_model}

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


def coders2transf():
    # print names of weights
    print([w.name for w in transformer.weights])
    print(len(transformer.weights))

    # same for the encoder model
    print([w.name for w in enc.weights])
    print(len(enc.weights))

    # print(transformer.weights[1])
    t_weights = transformer.weights
    print('here!')
    print(t_weights[0])
    print(t_weights[1])
    print(t_weights[2])
    t_names = [w.name for w in transformer.weights]

    for coder in coders.keys():
        encoders = np.unique([w.name.split('/')[0] for w in transformer.weights if coder in w.name])
        print(encoders)
        for i, e in enumerate(encoders):
            print(f'------------ {coder}', i, e)
            enl = [j for j, w in enumerate(transformer.weights) if e in w.name]
            print(enl)
            shuffled_coder = []
            for w in coders[coder].weights:
                wn = w.numpy()
                np.random.shuffle(wn)
                shuffled_coder.append(wn)

            for j, l in enumerate(enl):
                t_weights[l] = shuffled_coder[j]

    print('here!')
    print(t_weights[0])
    print(t_weights[1])
    print(t_weights[2])

    transformer.set_weights(t_weights)


optimizer = AdamW(learning_rate=lr, weight_decay=1e-4)

for e in range(epochs):
    # calculate the gradient
    with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
        # pass a random input to the encoder
        input_layer = tf.random.uniform((batch_size, SEQ_MAX_LEN_SOURCE, d_model))

        inp = tf.reshape(input_layer, (batch_size, -1))
        reshaped_inp = tf.reshape(inp, (batch_size, SEQ_MAX_LEN_SOURCE, d_model))
        mask_layer = tf.cast(tf.random.uniform((batch_size, 1, 1, SEQ_MAX_LEN_SOURCE)) > 0.5, tf.float32)
        output = enc_model([input_layer, mask_layer])

        # inp = input_layer
        # calculate the loss
        if 'meanaxis' in comments:
            shape = output.shape
            ones = np.array(shape) == 1
            deslice_axis = list(range(len(shape)))
            deslice_axis = [a for a, b in zip(deslice_axis, ones) if b == False and not a == 0]

            np.random.shuffle(deslice_axis)
            deslice_axis = deslice_axis[:-1]
            oup = output
            output = tf.reduce_mean(oup, axis=deslice_axis)

        norms, iloss, naswot_score = get_norms(tape, [inp], [output], comments=comments)
        loss = iloss
    print(e)
    grads = tape.gradient(loss, enc_model.trainable_weights)
    print(loss)
    print(grads)
    optimizer.apply_gradients(zip(grads, enc_model.trainable_weights))
