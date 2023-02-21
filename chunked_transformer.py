import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from alif_sg.neural_models.recLSC import get_norms
from alif_sg.neural_models.transformer_model import EncoderLayer, DecoderLayer
from alif_sg.neural_models.transformer_model import build_model
from tensorflow_addons.optimizers import AdamW

SEQ_MAX_LEN_SOURCE = 3
SEQ_MAX_LEN_TARGET = SEQ_MAX_LEN_SOURCE + 1
BPE_VOCAB_SIZE = 2
encoder_count = 2
decoder_count = encoder_count
batch_size = 32
attention_head_count = 2
d_model = 64
d_point_wise_ff = 2 * d_model
dropout_prob = .2
activation = 'swish'
comments = 'meanaxis_supsubnpsd_deslonly:1'
# comments = 'meanaxis_radius_deslonly:1'
# comments = 'meanaxis_radius'
# comments = 'supsubnpsd'
layer_index = 0
epochs = 20

if not 'supsubnpsd' in comments:
    lr = 1e-1  # 1e-6 went to 0.61 norm, 1e-3/1e-2 show an upper trend
    weight_decay = 1e-2

    optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
else:
    lr = 1e-2
    weight_decay = 1e-6

    # optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

rep_shape = (batch_size, SEQ_MAX_LEN_SOURCE, d_model)
mask_shape = [rep_shape[0]] + [1, 1] + [rep_shape[1]]


def get_enc_model():
    enc = EncoderLayer(
        attention_head_count,
        d_model,
        d_point_wise_ff,
        dropout_prob,
        activation=activation,
        comments=comments,
        layer_index=layer_index
    )

    input_layer = tf.keras.layers.Input(rep_shape[1:])
    mask_layer = tf.keras.layers.Input(mask_shape[1:])

    output = enc([input_layer, mask_layer])
    enc_model = tf.keras.Model(inputs=[input_layer, mask_layer], outputs=output)
    return enc_model


def get_dec_model():
    dec = DecoderLayer(
        attention_head_count,
        d_model,
        d_point_wise_ff,
        dropout_prob,
        activation=activation,
        comments=comments,
        layer_index=layer_index
    )
    decoder_tensor = tf.keras.layers.Input(rep_shape[1:])
    encoder_tensor = tf.keras.layers.Input(rep_shape[1:])
    target_padding_mask = tf.keras.layers.Input(mask_shape[1:])
    output = dec([decoder_tensor, encoder_tensor, target_padding_mask])
    dec_model = tf.keras.Model(inputs=[decoder_tensor, encoder_tensor, target_padding_mask], outputs=output)
    return dec_model


enc_model = get_enc_model()
dec_model = get_dec_model()

# here (16, 100, 512) (16, 1, 1, 100)


# get weights
enc_weights = enc_model.get_weights()

# turn weights to ones
enc_weights = [np.ones_like(w) for w in enc_weights]

# set weights
enc_model.set_weights(enc_weights)

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

coders = {
    'encoder': get_enc_model(),
    # 'decoder': get_dec_model()
}

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



n_inputs = {'encoder': 1, 'decoder': 2}

for coder_name in coders.keys():
    print(f'------------ {coder_name}')
    weights = None
    loss_list, norm_list = [], []
    for e in range(epochs):
        # calculate the gradient
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:

            coder_model = coders[coder_name]
            if not weights is None:
                coder_model.set_weights(weights)

            # pass a random input to the encoder
            inputs = []
            for _ in range(n_inputs[coder_name]):
                input_layer = tf.random.normal((batch_size, SEQ_MAX_LEN_SOURCE, d_model))
                tape.watch(input_layer)
                inputs.append(input_layer)

            inp = tf.reshape(input_layer, (batch_size, -1))
            reshaped_inp = tf.reshape(inp, (batch_size, SEQ_MAX_LEN_SOURCE, d_model))
            inputs[0] = reshaped_inp
            np.random.shuffle(inputs)

            mask_layer = tf.cast(tf.random.uniform((batch_size, 1, 1, SEQ_MAX_LEN_SOURCE)) > 0.5, tf.float32)
            tape.watch(mask_layer)

            output = coder_model([*inputs, mask_layer])

            # calculate the loss
            if 'meanaxis' in comments:
                if 'deslonly:1' in comments:
                    deslice_axis = [1]
                else:
                    shape = output.shape
                    ones = np.array(shape) == 1
                    deslice_axis = list(range(len(shape)))
                    deslice_axis = [d for d, o in zip(deslice_axis, ones) if o == False and not d == 0]

                    np.random.shuffle(deslice_axis)
                    deslice_axis = deslice_axis[:-1]
                output = tf.reduce_mean(output, axis=deslice_axis)
            norms, iloss, _ = get_norms(tape, [inp], [output], comments=comments)
            loss = iloss
        grads = tape.gradient(loss, coder_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, coder_model.trainable_weights))
        norm = tf.reduce_mean(norms).numpy().round(4)
        loss = tf.reduce_mean(loss).numpy().round(4)

        loss_list.append(loss)
        norm_list.append(norm)

        weights = coder_model.get_weights()

        # if not 'supsubnpsd' in comments:
        for w in weights:
            np.random.shuffle(w)

        print(f'Epoch {e} - loss: {str(loss)} - norm: {str(norm)} - desliced on {deslice_axis}')

    # plot norms and losses as subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(loss_list)
    ax1.set_title('Loss')
    ax2.plot(norm_list)
    ax2.set_title('Norm')
    plt.show()
