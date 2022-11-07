import os, time
import numpy as np
import warnings
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '1'
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from tqdm import tqdm
from GenericTools.keras_tools.esoteric_losses import well_loss
from GenericTools.keras_tools.esoteric_tasks.time_task_redirection import Task
from sg_design_lif.neural_models.full_model import build_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '1'
warnings.filterwarnings('ignore')


def get_norms(tape, lower_states, upper_states, n_samples=-1, norm_pow=2):
    hss = []
    for hlm1 in lower_states:
        hs = [tape.batch_jacobian(hl, hlm1) for hl in upper_states]
        hss.append(tf.concat(hs, axis=1))

    if len(hss) > 1:
        td = tf.concat(hss, axis=2)
    else:
        td = hss[0]

    del hss, hs

    # print('nonzero:', tf.math.count_nonzero(td))
    if n_samples < 1 and norm_pow in [1, 2, np.inf]:
        if norm_pow is np.inf:
            norms = tf.reduce_sum(tf.abs(td), axis=2)
            norms = tf.reduce_max(norms, axis=-1)

        elif norm_pow == 1:
            norms = tf.reduce_sum(tf.abs(td), axis=1)
            norms = tf.reduce_max(norms, axis=-1)

        elif norm_pow == 2:
            s, _, _ = tf.linalg.svd(td)
            norms = s[..., 0]

        else:
            raise NotImplementedError
    else:

        x = tf.random.normal((td.shape[0], td.shape[-1], n_samples))
        x_norm = tf.norm(x, ord=norm_pow, axis=1)
        e = tf.einsum('bij,bjk->bik', td, x)
        e_norm = tf.norm(e, ord=norm_pow, axis=1)

        norms = e_norm / x_norm
        norms = tf.reduce_max(norms, axis=-1)
    return norms


def apply_LSC(train_task_args, model_args, norm_pow, n_samples, batch_size, steps_per_epoch=2, epsilon=.01,
              patience=50, depth_norm=True, encoder_norm=False, decoder_norm=True, learn=True, time_steps=None,
              weights=None, save_weights_path=None):
    # FIXME: generalize this loop for any recurrent model
    gen_train = Task(**train_task_args)

    comments = model_args['comments']
    model_args['initial_state'] = ''

    stack = model_args['stack']
    net_name = model_args['net_name']
    if isinstance(model_args['stack'], str):
        stack = [int(s) for s in model_args['stack'].split(':')]
    elif isinstance(model_args['stack'], int):
        stack = [model_args['n_neurons'] for _ in range(model_args['stack'])]

    # batch = [tf.convert_to_tensor(tf.cast(b, tf.float32), dtype=tf.float32) for b in batch[0]],
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    states = []

    losses = []
    all_norms = []
    rec_norms = {}

    # the else is valid for the LSTM
    hi, ci = (1, 2) if 'LSNN' in net_name else (0, 1)
    n_states = 4 if 'LSNN' in net_name else 2

    for width in stack:
        for _ in range(n_states):
            states.append(tf.zeros((batch_size, width)))

    pbar1 = tqdm(total=steps_per_epoch, position=1)
    epsilon_steps = 0
    results = {}

    # get initial values of model
    model = build_model(**model_args)

    if not save_weights_path is None:
        os.makedirs(save_weights_path, exist_ok=True)
        # Guardar configuración JSON en el disco
        config_path = os.path.join(save_weights_path, 'model_config_lsc_before.json')
        json_config = model.to_json()
        with open(config_path, 'w') as json_file:
            json_file.write(json_config)
        # Guardar pesos en el disco
        weights_path = os.path.join(save_weights_path, 'model_weights_lsc_before.h5')
        model.save_weights(weights_path)

    if weights is None:
        weights = model.get_weights()
    weight_names = [weight.name for layer in model.layers for weight in layer.weights]
    results.update({f'{n}_mean': [tf.reduce_mean(w).numpy()] for n, w in zip(weight_names, weights)})
    results.update({f'{n}_var': [tf.math.reduce_variance(w).numpy()] for n, w in zip(weight_names, weights)})


    model, tape = None, None

    for step in range(steps_per_epoch):
        for i, _ in enumerate(stack):
            rec_norms[f'batch {step} layer {i}'] = []

        batch = gen_train.__getitem__(step)
        batch = [tf.convert_to_tensor(tf.cast(b, tf.float32), dtype=tf.float32) for b in batch[0]],
        if 'gausslsc' in comments:
            batch = [tf.convert_to_tensor(tf.cast(b, tf.float32), dtype=tf.float32) for b in batch[0]],
        elif 'berlsc' in comments:
            batch = [tf.math.greater(tf.random.uniform(b.shape), .5) for b in batch[0]],
            batch = [tf.cast(b, dtype=tf.float32) for b in batch[0]],
        elif 'shufflelsc' in comments:
            shapes = [b.shape for b in batch[0]]
            flatbatch = [tf.reshape(b, [-1, 2]) for b in batch[0]],
            shuffled = [tf.random.shuffle(b) for b in flatbatch[0]],
            batch = [tf.reshape(b, s) for s, b in zip(shapes, shuffled[0])],

        ts = batch[0][0].shape[1] if time_steps is None else time_steps
        pbar2 = tqdm(total=ts, position=0)

        for t in range(ts):
            bt = batch[0][0][:, t, :][:, None]
            wt = batch[0][1][:, t][:, None]

            del model, tape
            with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
                tape.watch(wt)
                tape.watch(bt)
                tape.watch(states)
                model = build_model(**model_args)
                model.set_weights(weights)

                outputs = model([bt, wt, *states])
                states_p1 = outputs[1:]

                mean_loss = 0
                some_norms = []
                state_below = None
                for i, _ in enumerate(stack):

                    htp1 = states_p1[i * n_states + hi]
                    ht = states[i * n_states + hi]
                    ctp1 = states_p1[i * n_states + ci]
                    ct = states[i * n_states + ci]

                    norms = get_norms(tape=tape, lower_states=[ht, ct], upper_states=[htp1, ctp1], n_samples=n_samples,
                                      norm_pow=norm_pow)
                    rec_norms[f'batch {step} layer {i}'].append(norms.numpy())
                    some_norms.append(tf.reduce_mean(norms))
                    loss = well_loss(min_value=1, max_value=1, walls_type='relu', axis='all')(norms)
                    mean_loss += loss

                    if encoder_norm and i == 0:
                        norms = get_norms(tape=tape, lower_states=[bt[:, 0, :]], upper_states=[htp1, ctp1],
                                          n_samples=n_samples,
                                          norm_pow=norm_pow)

                        some_norms.append(tf.reduce_mean(norms))
                        loss = well_loss(min_value=1, max_value=1, walls_type='relu', axis='all')(norms)
                        mean_loss += loss

                    if depth_norm:

                        hl = htp1
                        cl = ctp1
                        if not state_below is None:
                            hlm1, clm1 = state_below
                            norms = get_norms(tape=tape, lower_states=[hlm1, clm1], upper_states=[hl, cl],
                                              n_samples=n_samples, norm_pow=norm_pow)

                            some_norms.append(tf.reduce_mean(norms))
                            loss = well_loss(min_value=1, max_value=1, walls_type='relu', axis='all')(norms)
                            mean_loss += loss

                        state_below = (hl, cl)
                        del hl, cl

                    if decoder_norm and i == len(stack) - 1:
                        norms = get_norms(tape=tape, lower_states=[htp1, ctp1], upper_states=[outputs[0][:, 0, :]],
                                          n_samples=n_samples,
                                          norm_pow=norm_pow)

                        some_norms.append(tf.reduce_mean(norms))
                        loss = well_loss(min_value=1, max_value=1, walls_type='relu', axis='all')(norms)
                        mean_loss += loss

                    del htp1, ht, ctp1, ct

            if learn:
                grads = tape.gradient(mean_loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                del grads

            states = states_p1

            if not np.isnan(mean_loss.numpy()):
                del weights
                weights = model.get_weights()

            tf.keras.backend.clear_session()
            # print(some_norms)
            norms = tf.reduce_mean(some_norms)

            if abs(norms.numpy() - 1) < epsilon:
                epsilon_steps += 1

            if epsilon_steps > patience:
                break

            all_norms.append(norms.numpy())
            losses.append(mean_loss.numpy())

            pbar2.update(1)
            pbar2.set_description(
                f"Step {step}; "
                f"Loss {str(round(mean_loss.numpy(), 3))}; "
                f"mean params {str(round(tf.reduce_mean([tf.reduce_mean(w) for w in model.trainable_weights]).numpy(), 3))}; "
                f"mean norms {str(round(norms.numpy(), 3))} "
            )
            for n, w in zip(weight_names, model.get_weights()):
                results[f'{n}_mean'].append(tf.reduce_mean(w).numpy())
                results[f'{n}_var'].append(tf.math.reduce_variance(w).numpy())

        del batch

        pbar1.update(1)
        if epsilon_steps > patience:
            break

    del gen_train

    for n in weight_names:
        results[f'{n}_mean'] = str(results[f'{n}_mean'])
        results[f'{n}_var'] = str(results[f'{n}_var'])


    if not save_weights_path is None:
        # Guardar configuración JSON en el disco
        config_path = os.path.join(save_weights_path, 'model_config_lsc_after.json')
        json_config = model.to_json()
        with open(config_path, 'w') as json_file:
            json_file.write(json_config)
        # Guardar pesos en el disco
        weights_path = os.path.join(save_weights_path, 'model_weights_lsc_after.h5')
        model.save_weights(weights_path)

    del model, tape
    # results.update(LSC_losses=str(losses), LSC_norms=str(all_norms))
    results.update(LSC_losses=str(losses), LSC_norms=str(all_norms), rec_norms=rec_norms)

    # print(rec_norms)
    tf.keras.backend.clear_session()
    return weights, results


if __name__ == '__main__':

    FILENAME = os.path.realpath(__file__)
    CDIR = os.path.dirname(FILENAME)
    DATA = os.path.join(CDIR, 'data', )
    EXPERIMENTS = os.path.join(CDIR, 'experiments')
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)
    random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
    EXPERIMENT = os.path.join(EXPERIMENTS, time_string + random_string + '_normM')
    MODL = os.path.join(EXPERIMENT, 'trained_models')
    # GENDATA = os.path.join(DATA, 'initconds')

    for d in [EXPERIMENT, MODL]:
        os.makedirs(d, exist_ok=True)

    np.random.seed(42)
    tf.random.set_seed(42)

    input_dim = 2
    time_steps = 2
    batch_size = 2
    units = 4
    norm_pow = 0.1  # np.inf
    n_samples = 11
    lr = 1e-2

    stack = 2
    net_name = 'LSTM'  # maLSNN LSTM
    comments = ''

    comments += '_**folder:' + EXPERIMENT + '**_'
    comments += '_batchsize:' + str(batch_size)

    gen_train = Task(timerepeat=2, epochs=2, batch_size=batch_size, steps_per_epoch=2,
                     name='heidelberg', train_val_test='train', maxlen=100, comments=comments)

    comments += '_reoldspike'
    model_args = dict(task_name='heidelberg', net_name=net_name, n_neurons=units, lr=lr, stack=stack,
                      loss_name='sparse_categorical_crossentropy', tau=1., tau_adaptation=1.,
                      embedding=None, optimizer_name='AdaBelief', lr_schedule='',
                      weight_decay=1, clipnorm=1, initializer='glorot_uniform', comments=comments,
                      in_len=gen_train.in_len, n_in=gen_train.in_dim, out_len=gen_train.out_len,
                      n_out=gen_train.out_dim, final_epochs=1)
    # model = build_model(**model_args)

    weights, losses, all_norms = apply_LSC(gen_train, model_args, norm_pow, n_samples)
    plt.plot(losses)
    plt.show()
