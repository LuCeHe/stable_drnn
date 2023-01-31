import os

import numpy as np
import tensorflow as tf

from GenericTools.keras_tools.esoteric_layers import Identity
from GenericTools.keras_tools.esoteric_models.model import modifiedModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Transformer(object):
    def __init__(self,
                 inputs_vocab_size,
                 target_vocab_size,
                 encoder_count,
                 decoder_count,
                 attention_head_count,
                 d_model,
                 d_point_wise_ff,
                 dropout_prob, activation='relu', comments=''):

        # model hyper parameter variables
        self.encoder_count = encoder_count
        self.decoder_count = decoder_count
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob
        self.comments = comments

        self.encoder_embedding_layer = Embeddinglayer(inputs_vocab_size, d_model)
        self.encoder_embedding_dropout = tf.keras.layers.Dropout(dropout_prob)

        if not 'sameemb' in comments:
            self.decoder_embedding_layer = Embeddinglayer(target_vocab_size, d_model)
        else:
            self.decoder_embedding_layer = self.encoder_embedding_layer
        self.decoder_embedding_dropout = tf.keras.layers.Dropout(dropout_prob)

        self.decoder_embedding_layer.embedding.build(None)
        self.emb_matrix = tf.transpose(self.decoder_embedding_layer.embedding.embeddings)

        self.encoder_layers = [
            EncoderLayer(
                attention_head_count,
                d_model,
                d_point_wise_ff,
                dropout_prob,
                activation=activation,
                comments=comments,
                layer_index=i
            ) for i in range(encoder_count)
        ]

        self.decoder_layers = [
            DecoderLayer(
                attention_head_count,
                d_model,
                d_point_wise_ff,
                dropout_prob,
                activation=activation,
                comments=comments,
                layer_index=i
            ) for i in range(decoder_count)
        ]

        # self.linear = tf.keras.layers.Dense(target_vocab_size)

    def __call__(self, inputs):
        source, target, inputs_padding_mask, look_ahead_mask, target_padding_mask = inputs
        # print(inputs.shape, target.shape, inputs_padding_mask.shape, look_ahead_mask.shape, target_padding_mask.shape)
        encoder_tensor = self.encoder_embedding_layer(source)
        encoder_tensor = self.encoder_embedding_dropout(encoder_tensor)
        target = self.decoder_embedding_layer(target)
        decoder_tensor = self.decoder_embedding_dropout(target)

        encoded_ts = []
        for i in range(self.encoder_count):
            encoder_tensor = self.encoder_layers[i]([encoder_tensor, inputs_padding_mask])
            encoded_ts.append(encoder_tensor)

        for i in range(self.decoder_count):
            decoder_tensor = self.decoder_layers[i]([decoder_tensor, encoder_tensor, look_ahead_mask,
                                                          target_padding_mask])

        output = decoder_tensor @ self.emb_matrix
        return output


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob, activation='relu', comments='',
                 layer_index=0, **kwargs):
        super().__init__(**kwargs)

        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob
        self.layer_index = layer_index

        self.attention = MultiHeadAttention(attention_head_count, d_model)

        self.dropout_1 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
            d_point_wise_ff, d_model, activation=activation
        )
        self.dropout_2 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.id1 = Identity(name=f'eidentity_1_{layer_index}')
        self.id2 = Identity(name=f'eidentity_2_{layer_index}')
        self.id3 = Identity(name=f'eidentity_3_{layer_index}')

        self.add1 = tf.keras.layers.Add(name=f'eadd_1_{layer_index}')
        self.add2 = tf.keras.layers.Add(name=f'eadd_2_{layer_index}')

    def call(self, inputs, **kwargs):
        x, mask = inputs
        mask = tf.stop_gradient(mask)

        x = self.id1(x)
        output, attention = self.attention([x, x, x, mask])
        output = self.dropout_1(output)
        output = self.layer_norm_1(self.add1([x, output]))  # residual network
        output = self.id2(output)
        output_temp = output

        output = self.position_wise_feed_forward_layer(output)
        output = self.dropout_2(output)
        output = self.layer_norm_2(self.add2([output_temp, output]))
        output = self.id3(output)

        return output


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob, activation='relu', comments='',
                 layer_index=0, **kwargs):
        super().__init__(**kwargs)

        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob
        self.layer_index = layer_index

        self.attention = MultiHeadAttention(attention_head_count, d_model)

        self.dropout_1 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.conditioned_attention = MultiHeadAttention(attention_head_count, d_model)

        self.dropout_2 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
            d_point_wise_ff, d_model, activation=activation
        )
        self.dropout_3 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.id1 = Identity(name=f'didentity_1_{layer_index}')
        self.id2 = Identity(name=f'didentity_2_{layer_index}')
        self.id3 = Identity(name=f'didentity_3_{layer_index}')

        self.add1 = tf.keras.layers.Add(name=f'dadd_1_{layer_index}')
        self.add2 = tf.keras.layers.Add(name=f'dadd_2_{layer_index}')
        self.add3 = tf.keras.layers.Add(name=f'dadd_3_{layer_index}')

    def call(self, inputs, **kwargs):
        decoder_inputs, encoder_output, look_ahead_mask, padding_mask = inputs
        # stop gradients through look_ahead_mask
        look_ahead_mask = tf.stop_gradient(look_ahead_mask)
        padding_mask = tf.stop_gradient(padding_mask)

        encoder_length = tf.shape(encoder_output)[1]
        # concatenate encode and decoder representations on the time axis
        concats = tf.concat([encoder_output, decoder_inputs], axis=1)

        concats = self.id1(concats)

        # deconcatenate representations
        encoder_output = concats[:, :encoder_length, :]
        decoder_inputs = concats[:, encoder_length:, :]

        output, attention_1 = self.attention([decoder_inputs, decoder_inputs, decoder_inputs, look_ahead_mask])
        output = self.dropout_1(output)
        query = self.layer_norm_1(self.add1([decoder_inputs, output]))  # residual network
        output, attention_2 = self.conditioned_attention([query, encoder_output, encoder_output, padding_mask])
        output = self.dropout_2(output)
        encoder_decoder_attention_output = self.layer_norm_2(self.add2([output, query]))
        encoder_decoder_attention_output = self.id1(encoder_decoder_attention_output)

        output = self.position_wise_feed_forward_layer(encoder_decoder_attention_output)
        output = self.dropout_3(output)
        output = self.layer_norm_3(self.add3([encoder_decoder_attention_output, output]))  # residual network
        output = self.id3(output)

        return output


class PositionWiseFeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, d_point_wise_ff, d_model, activation='relu'):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.w_1 = tf.keras.layers.Dense(d_point_wise_ff)
        self.w_2 = tf.keras.layers.Dense(d_model)
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, inputs, **kwargs):
        inputs = self.w_1(inputs)
        inputs = self.activation(inputs)
        return self.w_2(inputs)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model):
        super(MultiHeadAttention, self).__init__()

        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model

        if d_model % attention_head_count != 0:
            raise ValueError(
                "d_model({}) % attention_head_count({}) is not zero.d_model must be multiple of attention_head_count.".format(
                    d_model, attention_head_count
                )
            )

        self.d_h = d_model // attention_head_count

        self.w_query = tf.keras.layers.Dense(d_model, use_bias=False)
        self.w_key = tf.keras.layers.Dense(d_model, use_bias=False)
        self.w_value = tf.keras.layers.Dense(d_model, use_bias=False)

        self.scaled_dot_product = ScaledDotProductAttention(self.d_h)

        self.ff = tf.keras.layers.Dense(d_model, use_bias=False)

    def call(self, inputs, **kwargs):
        query, key, value, mask = inputs
        batch_size = tf.shape(query)[0]

        query = self.w_query(query)
        key = self.w_key(key)
        value = self.w_value(value)

        query = self.split_head(query, batch_size)
        key = self.split_head(key, batch_size)
        value = self.split_head(value, batch_size)

        output, attention = self.scaled_dot_product(query, key, value, mask)
        output = self.concat_head(output, batch_size)

        return self.ff(output), attention

    def split_head(self, tensor, batch_size):
        # inputs tensor: (batch_size, seq_len, d_model)
        return tf.transpose(
            tf.reshape(
                tensor,
                (batch_size, -1, self.attention_head_count, self.d_h)
                # tensor: (batch_size, seq_len_splited, attention_head_count, d_h)
            ),
            [0, 2, 1, 3]
            # tensor: (batch_size, attention_head_count, seq_len_splited, d_h)
        )

    def concat_head(self, tensor, batch_size):
        return tf.reshape(
            tf.transpose(tensor, [0, 2, 1, 3]),
            (batch_size, -1, self.attention_head_count * self.d_h)
        )


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_h):
        super(ScaledDotProductAttention, self).__init__()
        self.d_h = d_h

    def call(self, query, key, value, mask=None):
        matmul_q_and_transposed_k = tf.matmul(query, key, transpose_b=True)
        scale = tf.sqrt(tf.cast(self.d_h, dtype=tf.float32))
        scaled_attention_score = matmul_q_and_transposed_k / scale
        if mask is not None:
            scaled_attention_score += (mask * -1e9)

        attention_weight = tf.nn.softmax(scaled_attention_score, axis=-1)

        return tf.matmul(attention_weight, value), attention_weight


class Embeddinglayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        # model hyper parameter variables
        super(Embeddinglayer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

    def call(self, sequences, **kwargs):
        # print(sequences)
        if isinstance(sequences, list):
            # fixme:
            sequences = sequences[0]

        max_sequence_len = sequences.shape[1]
        output = self.embedding(sequences) * tf.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        output += self.positional_encoding(max_sequence_len)

        return output

    def positional_encoding(self, max_len):
        pos = np.expand_dims(np.arange(0, max_len), axis=1)
        index = np.expand_dims(np.arange(0, self.d_model), axis=0)

        pe = self.angle(pos, index)

        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        pe = np.expand_dims(pe, axis=0)
        return tf.cast(pe, dtype=tf.float32)

    def angle(self, pos, index):
        return pos / np.power(10000, (index - index % 2) / np.float32(self.d_model))


def build_model(
        inputs_timesteps,
        target_timesteps,
        inputs_vocab_size,
        target_vocab_size,
        encoder_count,
        decoder_count,
        attention_head_count,
        d_model,
        d_point_wise_ff,
        dropout_prob,
        activation='relu',
        comments='',
):
    transformer = Transformer(
        inputs_vocab_size=inputs_vocab_size,
        target_vocab_size=target_vocab_size,
        encoder_count=encoder_count,
        decoder_count=decoder_count,
        attention_head_count=attention_head_count,
        d_model=d_model,
        d_point_wise_ff=d_point_wise_ff,
        dropout_prob=dropout_prob,
        activation=activation,
        comments=comments
    )

    # print(transformer.inputs)
    # print(transformer.outputs)

    inputs_layer = tf.keras.layers.Input((inputs_timesteps,))
    target_layer = tf.keras.layers.Input((target_timesteps - 1,))
    inputs_padding_mask = tf.keras.layers.Input((1, 1, inputs_timesteps,))
    look_ahead_mask = tf.keras.layers.Input((1, inputs_timesteps, inputs_timesteps,))
    target_padding_mask = tf.keras.layers.Input((1, 1, target_timesteps - 1,))

    output = transformer([inputs_layer, target_layer, inputs_padding_mask, look_ahead_mask, target_padding_mask])

    model = modifiedModel(
        [inputs_layer, target_layer, inputs_padding_mask, look_ahead_mask, target_padding_mask],
        output,
        name='Transformer')

    # model = tf.keras.models.Model(
    #     [inputs_layer, target_layer, inputs_padding_mask, look_ahead_mask, target_padding_mask],
    #     output
    # )

    # model.summary()
    return model
