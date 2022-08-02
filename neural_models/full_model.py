from tensorflow.keras.layers import *
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.losses import sparse_categorical_crossentropy

from GenericTools.keras_tools.esoteric_layers import *
from GenericTools.keras_tools.esoteric_layers.dropin import DropIn
from GenericTools.keras_tools.esoteric_optimizers.optimizer_selection import get_optimizer
from GenericTools.stay_organized.utils import str2val
from GenericTools.keras_tools.esoteric_losses.loss_redirection import get_loss
from GenericTools.keras_tools.esoteric_losses.advanced_losses import *

import alif_sg.neural_models as models
from alif_sg.neural_models.custom_lstm import customLSTMcell

metrics = [
    sparse_categorical_accuracy,
    bpc,
    perplexity,
    # tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    sparse_mode_accuracy,
    sparse_categorical_crossentropy,
]


def Expert(i, j, stateful, task_name, net_name, n_neurons, initializer, comments, batch_size):
    ij = '_{}_{}'.format(i, j)

    thr = str2val(comments, 'thr', float, .01)

    if 'convWin' in comments:
        kernel_size = str2val(comments, 'ksize', int, 4)
        win = Conv1D(filters=int(n_neurons), kernel_size=kernel_size, dilation_rate=1, padding='causal')
    else:
        win = lambda x: x

    batch_size = str2val(comments, 'batchsize', int, batch_size)
    maxlen = str2val(comments, 'maxlen', int, 100)
    nin = str2val(comments, 'nin', int, 1) if not 'convWin' in comments else n_neurons

    if 'LSNN' in net_name:
        stack_info = '_stacki:{}'.format(i)
        cell = models.net(net_name)(
            num_neurons=n_neurons, initializer=initializer, config=comments + stack_info, thr=thr)
        rnn = RNN(cell, return_sequences=True, name='encoder' + ij, stateful=stateful)

    elif 'cLSTM' in net_name:
        cell = customLSTMcell(num_neurons=n_neurons, string_config=comments, name=f'alsnn_{i}')
        rnn = RNN(cell, return_sequences=True, name='encoder' + ij, stateful=stateful)
    else:
        raise NotImplementedError

    rnn.build((batch_size, maxlen, nin))

    def call(inputs):
        skipped_connection_input, output_words = inputs
        skipped_connection_input = win(skipped_connection_input)
        if 'LSNN' in net_name:
            b, v, thr, v_sc = rnn(inputs=skipped_connection_input)

            if 'readout_voltage' in comments:
                output_cell = v
            else:
                output_cell = b
        else:
            output_cell = rnn(inputs=skipped_connection_input)[0]
        return output_cell

    return call


def build_model(task_name, net_name, n_neurons, lr, stack, batch_size,
                loss_name, embedding, optimizer_name, lr_schedule, weight_decay, clipnorm,
                initializer, comments, in_len, n_in, out_len, n_out, final_epochs,
                language_tasks, stateful=None):
    drate = str2val(comments, 'dropout', float, .1)
    # network definition
    # weights initialization
    embedding = embedding if task_name in language_tasks else False
    stateful = True if 'ptb' in task_name else False

    if 'stateful' in comments: stateful = True
    if stateful is False: stateful = False
    loss = get_loss(loss_name)

    # n_experts = str2val(comments, 'experts', int, 1)

    if not embedding is False:
        emb = SymbolAndPositionEmbedding(
            maxlen=in_len, vocab_size=n_out, embed_dim=n_neurons, embeddings_initializer=initializer,
            from_string=embedding)

        emb.build((1, n_out))
        emb.sym_emb.build((1, n_out))
        mean = np.mean(np.mean(emb.sym_emb.embeddings, axis=-1), axis=-1)
        var = np.mean(np.var(emb.sym_emb.embeddings, axis=-1), axis=-1)
        comments = str2val(comments, 'taskmean', replace=mean)
        comments = str2val(comments, 'taskvar', replace=var)
        comments = str2val(comments, 'embdim', replace=emb.embed_dim)

    # graph
    input_words = Input([None, n_in], name='input_spikes')
    output_words = Input([None], name='target_words')

    x = input_words

    if not embedding is False:
        if x.shape[-1] == 1:
            in_emb = Lambda(lambda z: tf.squeeze(z, axis=-1), name='Squeeze')(x)
        else:
            in_emb = Lambda(lambda z: tf.math.argmax(z, axis=-1), name='Argmax')(x)
        rnn_input = emb(in_emb)
    else:
        rnn_input = x  # [input_scaling * x]

    expert = lambda i, j, c, n: Expert(i, j, stateful, task_name, net_name, n_neurons=n,
                                       initializer=initializer, comments=c, batch_size=batch_size)

    if isinstance(stack, str):
        stack = [int(s) for s in stack.split(':')]
    elif isinstance(stack, int):
        stack = [n_neurons for _ in range(stack)]

    if 'preprocesstask' in task_name:
        tfe = tf.keras.layers.experimental.preprocessing
        rnn_input = ExpandDims(axis=-1)(rnn_input)
        rnn_input = tfe.RandomTranslation(.2, .2, fill_mode="wrap", interpolation="nearest")(rnn_input)
        rnn_input = tfe.RandomZoom(.2, .2, fill_mode="wrap", interpolation="nearest")(rnn_input)
        rnn_input = tfe.RandomRotation((-.1, .1), fill_mode="wrap", interpolation="nearest")(rnn_input)
        rnn_input = Squeeze(axis=-1)(rnn_input)
        rnn_input = DropIn(.3, binary=True)(rnn_input)

    for i, layer_width in enumerate(stack):
        rnn_input = Dropout(drate)(rnn_input)
        rnn_input = [rnn_input] if not isinstance(rnn_input, list) else rnn_input

        if i == 0:
            if not embedding is False:
                nin = emb.embed_dim
            else:
                nin = n_in
        else:
            nin = stack[i - 1]

        c = str2val(comments, '_nin', replace=nin)
        output_cell = expert(i, 0, c, n=layer_width)([rnn_input, output_words])
        rnn_input = output_cell

    output = output_cell

    if 'embproj' in comments:
        output_net = emb(output, mode='projection')
    elif 'nsLIFreadout' in comments:
        cell = models.non_spiking_LIF(num_neurons=n_neurons, initializer=initializer)
        readout = RNN(cell, return_sequences=True, name='decoder', stateful=stateful)
        output_net = readout(output)
    else:
        readout = Dense(n_out, name='decoder', kernel_initializer=initializer)
        output_net = readout(output)

    loss = str2val(comments, 'loss', output_type=str, default=loss)
    output_net = AddLossLayer(loss=loss)([output_words, output_net])
    output_net = AddMetricsLayer(metrics=metrics)([output_words, output_net])
    output_net = Lambda(lambda z: z, name='output_net')(output_net)

    # train model
    train_model = tf.keras.models.Model([input_words, output_words], output_net, name=net_name)
    exclude_from_weight_decay = ['decoder'] if 'dontdecdec' in comments else []

    optimizer_name = str2val(comments, 'optimizer', output_type=str, default=optimizer_name)
    lr_schedule = str2val(comments, 'lrs', output_type=str, default=lr_schedule)
    optimizer = get_optimizer(optimizer_name=optimizer_name, lr_schedule=lr_schedule,
                              total_steps=final_epochs, lr=lr, weight_decay=weight_decay,
                              clipnorm=clipnorm, exclude_from_weight_decay=exclude_from_weight_decay)
    train_model.compile(optimizer=optimizer, loss=lambda x, y: 0)

    return train_model
