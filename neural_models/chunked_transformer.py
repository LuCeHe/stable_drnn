from tqdm import tqdm

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from alif_sg.neural_models.recLSC import get_norms
from alif_sg.neural_models.transformer_model import EncoderLayer, DecoderLayer, Embeddinglayer
from alif_sg.neural_models.transformer_model import build_model
from tensorflow_addons.optimizers import AdamW

from GenericTools.stay_organized.utils import str2val


def get_enc_model(attention_head_count, d_model, d_point_wise_ff, dropout_prob, activation, comments, layer_index,
                  rep_shape, mask_shape):
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
    model = tf.keras.Model(inputs=[input_layer, mask_layer], outputs=output)
    return model


def get_emb_model(BPE_VOCAB_SIZE, d_model, rep_shape):
    emb = Embeddinglayer(BPE_VOCAB_SIZE, d_model)

    input_layer = tf.keras.layers.Input((rep_shape[1],))

    output = emb(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    return model


def get_dec_model(attention_head_count, d_model, d_point_wise_ff, dropout_prob, activation, comments, layer_index,
                  rep_shape, mask_shape):
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
    model = tf.keras.Model(inputs=[decoder_tensor, encoder_tensor, target_padding_mask], outputs=output)
    return model


def coders2transf(coders, encoder_count, decoder_count, attention_head_count, d_model, d_point_wise_ff, dropout_prob,
                  activation, comments, batch_size, SEQ_MAX_LEN_SOURCE, SEQ_MAX_LEN_TARGET, BPE_VOCAB_SIZE):
    print('Loading pretrained weights to the Transformer...')

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

    t_weights = transformer.weights

    for coder in tqdm(coders.keys()):
        encoders = np.unique([w.name.split('/')[0] for w in transformer.weights if coder in w.name])
        # print(encoders)
        for i, e in enumerate(encoders):
            # print(f'------------ {coder}', i, e)
            enl = [j for j, w in enumerate(transformer.weights) if e in w.name]
            # print(enl)
            shuffled_coder = []
            for w in coders[coder].weights:
                wn = w.numpy()
                np.random.shuffle(wn)
                shuffled_coder.append(wn)

            for j, l in enumerate(enl):
                # print(t_weights[l].shape, shuffled_coder[j].shape)
                t_weights[l] = shuffled_coder[j]

    print('Weights were passed to the Transformer!')
    transformer.set_weights(t_weights)
    return transformer


def chunked_lsc(
        SEQ_MAX_LEN_SOURCE=50,
        SEQ_MAX_LEN_TARGET=51,
        pretrain_SEQ_MAX_LEN_SOURCE=50,
        BPE_VOCAB_SIZE=200,
        encoder_count=2,
        decoder_count=2,
        batch_size=16,
        attention_head_count=2,
        d_model=256,
        d_point_wise_ff=1024,
        dropout_prob=.2,
        activation='swish',
        comments='chunked_deslice_findLSC_radius_meanaxis',
        # comments='chunked_deslice_findLSC_supsubnpsd_meanaxis_noimagloss_normri',
        plot_pretraining=False,
        layer_index=0,
        epochs=30,
):
    comments += '_deslonly:1'
    if 'radius' in comments:
        lr = 1e-1  # 1e-6 went to 0.61 norm, 1e-3/1e-2 show an upper trend
        weight_decay = 1e-5

        optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay)

    elif not 'supsubnpsd' in comments:
        lr = 1e-1  # 1e-6 went to 0.61 norm, 1e-3/1e-2 show an upper trend
        weight_decay = 1e-2

        optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay)
    else:
        lr = 1e-2  # 1e-6 went to 0.61 norm, 1e-3/1e-2 show an upper trend
        weight_decay = 1e-5

        optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay)

    rep_shape = (batch_size, pretrain_SEQ_MAX_LEN_SOURCE, d_model)
    mask_shape = (batch_size, 1, 1, pretrain_SEQ_MAX_LEN_SOURCE)

    coders_fn = {
        'encoder': get_enc_model(attention_head_count, d_model, d_point_wise_ff, dropout_prob, activation, comments,
                                 layer_index, rep_shape, mask_shape),
        'decoder': get_dec_model(attention_head_count, d_model, d_point_wise_ff, dropout_prob, activation, comments,
                                 layer_index, rep_shape, mask_shape),
        # 'embedding': get_emb_model(),
    }

    enc = get_enc_model(attention_head_count, d_model, d_point_wise_ff, dropout_prob, activation, comments, layer_index,
                        rep_shape, mask_shape)
    dec = get_dec_model(attention_head_count, d_model, d_point_wise_ff, dropout_prob, activation, comments, layer_index,
                        rep_shape, mask_shape)
    coders_2_transformer = {
        'encoder': enc,
        'decoder': dec,
    }
    n_inputs = {'encoder': 1, 'decoder': 2, 'embedding': 1}

    max_dim = str2val(comments, 'maxdim', int, default=1024)

    results = {}
    for coder_name in coders_fn.keys():
        print(f'------------ Pretraining the {coder_name}')
        weights = None
        loss_list, norm_list = [], []
        pbar = tqdm(total=epochs)
        target_shape = (batch_size, pretrain_SEQ_MAX_LEN_SOURCE, n_inputs[coder_name] * d_model)
        li, ni = None, None
        ma_loss, ma_norm = None, None
        for e in range(epochs):
            # if True:
            try:
                # calculate the gradient
                with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:

                    coder_model = coders_fn[coder_name]
                    if not weights is None:
                        coder_model.set_weights(weights)

                    # pass a random input to the encoder
                    inputs = []
                    # for _ in range(n_inputs[coder_name]):
                    inp = None
                    if not coder_name == 'embedding':
                        if 'sphere' in comments:
                            input_layer = tf.random.uniform(target_shape, minval=-1, maxval=1)
                            norm = tf.norm(input_layer, axis=-1, keepdims=True)
                            input_layer = input_layer / norm

                        elif 'deflect' in comments:
                            inp = tf.random.normal((batch_size, max_dim))
                            projector = tf.random.uniform([max_dim] + list(target_shape[1:]), minval=-1, maxval=1)

                            tape.watch(inp)
                            tape.watch(projector)

                            dest = ''.join(np.random.choice(list('klmnop'), size=len(target_shape[1:]), replace=False))
                            projection = tf.einsum(f'ij,j{dest}->i{dest}', inp, projector)
                            input_layer = projection
                        else:
                            input_layer = tf.random.uniform(target_shape, minval=-1, maxval=1)


                    else:
                        # random integers from 0 to 100
                        input_layer = tf.random.uniform(target_shape[:2], minval=0,
                                                        maxval=BPE_VOCAB_SIZE,
                                                        dtype=tf.int32)

                    tape.watch(input_layer)

                    if 'encoder' in coder_name:
                        inputs.append(input_layer)
                    else:
                        # split input_layer in 2 in the -1 axis
                        layers = tf.split(input_layer, 2, axis=-1)
                        inputs.extend(layers)
                        # input_layer = layers[0]

                    if inp is None:
                        inp = tf.reshape(input_layer, (batch_size, -1))
                        if not coder_name == 'embedding':
                            reshaped_inp = tf.reshape(inp, target_shape)
                        else:
                            reshaped_inp = inp

                        inputs[0] = reshaped_inp

                    if not coder_name == 'embedding':
                        mask_layer = tf.cast(tf.random.uniform((batch_size, 1, 1, pretrain_SEQ_MAX_LEN_SOURCE)) > 0.5,
                                             tf.float32)
                        tape.watch(mask_layer)
                        inputs.append(mask_layer)

                    output = coder_model(inputs)

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
                    else:
                        output = tf.reshape(output, (batch_size, -1))
                    norms, iloss, _ = get_norms(tape, [inp], [output], comments=comments, n_s=-1)
                    loss = iloss
                grads = tape.gradient(loss, coder_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, coder_model.trainable_weights))
                norm = tf.reduce_mean(norms).numpy()
                loss = tf.reduce_mean(loss).numpy()

                loss_list.append(loss)
                norm_list.append(norm)



                weights = coder_model.get_weights()

                if not 'noshuffle' in comments:
                    for w in weights:
                        np.random.shuffle(w)

                ma_loss = loss if ma_loss is None else ma_loss * 9 / 10 + loss / 10
                ma_norm = norm if ma_norm is None else ma_norm * 9 / 10 + norm / 10

                if li is None:
                    li = str(loss.round(4))
                    ni = str(norm.round(4))
                pbar.update(1)
                pbar.set_description(
                    f'Epoch {e} - loss: {str(ma_loss.round(4))}/{li} - norm: {str(ma_norm.round(4))}/{ni}'
                    f' - desliced on {deslice_axis}'
                )
            except Exception as e:
                print(e)
        if not weights is None:
            coders_2_transformer[coder_name].set_weights(weights)

        results[coder_name + '_loss'] = loss_list
        results[coder_name + '_norm '] = norm_list
        if plot_pretraining:
            # plot norms and losses as subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax1.plot(loss_list)
            ax1.set_title('Loss')
            ax2.plot(norm_list)
            ax2.set_title('Norm')
            plt.show()

    transformer = coders2transf(coders_2_transformer, encoder_count, decoder_count, attention_head_count, d_model,
                                d_point_wise_ff, dropout_prob, activation, comments, batch_size, SEQ_MAX_LEN_SOURCE,
                                SEQ_MAX_LEN_TARGET, BPE_VOCAB_SIZE)

    return transformer.get_weights(), results


if __name__ == '__main__':
    chunked_lsc()
