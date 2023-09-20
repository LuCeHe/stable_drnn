import os, time, shutil
import numpy as np
import warnings
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa

from alif_sg.tools.admin_model_removal import get_pretrained_file
from pyaromatics.keras_tools.esoteric_layers.linear_recurrent_unit import ResLRUCell
from pyaromatics.keras_tools.esoteric_optimizers.AdamW import AdamW as AdamW2

from pyaromatics.keras_tools.convenience_operations import sample_axis
from pyaromatics.keras_tools.esoteric_layers import AddLossLayer, AddMetricsLayer, SymbolAndPositionEmbedding
from pyaromatics.keras_tools.esoteric_layers.rate_voltage_reg import RateVoltageRegularization
from pyaromatics.keras_tools.learning_rate_schedules import DummyConstantSchedule
from pyaromatics.stay_organized.utils import str2val, timeStructured
from sg_design_lif.neural_models import maLSNN, maLSNNb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '1'
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from tqdm import tqdm
from pyaromatics.keras_tools.esoteric_losses import well_loss
from pyaromatics.keras_tools.esoteric_tasks.time_task_redirection import Task
from sg_design_lif.neural_models.full_model import build_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '1'
warnings.filterwarnings('ignore')

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
GEXPERIMENTS = os.path.abspath(os.path.join(CDIR, '..', 'good_experiments'))
os.makedirs(GEXPERIMENTS, exist_ok=True)


def get_norms(tape=None, lower_states=None, upper_states=None, n_samples=-1, norm_pow=2, naswot=0, comments='',
              log_epsilon=1e-8, target_norm=1., n_s=16, test=False):
    if tape is None and lower_states is None and upper_states is None and test == False:
        raise ValueError('No input data given!')

    norms = None
    loss = 0
    upper_states = [tf.squeeze(hl) for hl in upper_states]
    if not test:
        hss = []
        for hlm1 in lower_states:
            hs = [tape.batch_jacobian(hl, hlm1, experimental_use_pfor=True) for hl in upper_states]
            # hs_aux = [tape.batch_jacobian(hlm1,hl,  experimental_use_pfor=True) for hl in upper_states]
            print(tape.gradient(upper_states[0], hlm1))
            print(tape.gradient(hlm1, upper_states[0]))
            print(hs)
            # print(hs_aux)
            hss.append(tf.concat(hs, axis=1))

        if len(hss) > 1:
            td = tf.concat(hss, axis=2)
        else:
            td = hss[0]

        del hss, hs
    else:
        td = tape

    if td.shape[-1] == td.shape[-2]:
        std = td
    else:
        if tf.math.greater(td.shape[1], td.shape[2]):
            sample_ax = 1
            max_dim = td.shape[2]
        else:
            sample_ax = 2
            max_dim = td.shape[1]

        std = sample_axis(td, max_dim=max_dim, axis=sample_ax)

    if 'supnpsd' in comments:
        # loss that encourages the matrix to be psd
        z = tf.random.normal((n_s, std.shape[-1]))
        zn = tf.norm(z, ord='euclidean', axis=-1)
        z = z / tf.expand_dims(zn, axis=-1)
        zT = tf.transpose(z)

        a = std @ zT
        preloss = tf.einsum('bks,sk->bs', a, z)
        loss += tf.reduce_mean(tf.nn.relu(-preloss)) / 4

        # norms = tf.linalg.logdet(std) + 1
        norms = tf.reduce_sum(tf.math.log(log_epsilon + tf.abs(tf.linalg.eigvals(std))), axis=-1) + 1


    elif 'supsubnpsd' in comments:
        # loss that encourages the matrix to be psd
        if n_s > 0:
            z = tf.random.normal((n_s, std.shape[-1]))
            zn = tf.norm(z, ord='euclidean', axis=-1)
            z = z / tf.expand_dims(zn, axis=-1)
            zT = tf.transpose(z)

            a = std @ zT
            preloss = tf.einsum('bks,sk->bs', a, z)
            loss += tf.reduce_mean(tf.nn.relu(-preloss)) / 4

        eig = tf.linalg.eigvals(std)
        r = tf.math.real(eig)
        i = tf.math.imag(eig)

        if 'normri' in comments:
            norms = tf.math.sqrt(r ** 2 + i ** 2)
        else:
            norms = r

        if not 'noimagloss' in comments:
            loss += well_loss(min_value=0., max_value=0., walls_type='relu', axis='all')(i) / 4

    elif 'logradius' in comments:
        if td.shape[-1] == td.shape[-2]:
            r = tf.math.reduce_max(tf.abs(tf.linalg.eigvals(td)), axis=-1)
        else:
            r = tf.math.reduce_max(tf.linalg.svd(td, compute_uv=False), axis=-1) / 2
        norms = tf.math.log(r + log_epsilon) + 1

    elif 'radius' in comments:
        norms = tf.math.reduce_max(tf.abs(tf.linalg.eigvals(std)), axis=-1)

    elif 'entrywise' in comments:
        flat_td = tf.reshape(td, (td.shape[0], -1))
        norms = tf.norm(flat_td, ord=norm_pow, axis=1)

    elif n_samples < 1 and norm_pow in [1, 2, np.inf]:
        if norm_pow is np.inf:
            norms = tf.reduce_sum(tf.abs(td), axis=2)
            norms = tf.reduce_max(norms, axis=-1)

        elif norm_pow == 1:
            norms = tf.reduce_sum(tf.abs(td), axis=1)
            norms = tf.reduce_max(norms, axis=-1)

        elif norm_pow == 2:
            norms = tf.linalg.svd(td, compute_uv=False)[..., 0]

    elif n_samples > 0:
        x = tf.random.normal((td.shape[0], td.shape[-1], n_samples))
        x_norm = tf.norm(x, ord=norm_pow, axis=1)
        e = tf.einsum('bij,bjk->bik', td, x)
        e_norm = tf.norm(e, ord=norm_pow, axis=1)

        norms = e_norm / x_norm
        norms = tf.reduce_max(norms, axis=-1)

    else:
        norms = tf.math.reduce_max(tf.abs(tf.linalg.eigvals(std)), axis=-1)

    loss += well_loss(min_value=target_norm, max_value=target_norm, walls_type='squared', axis='all')(norms)

    return tf.abs(norms), loss, None


def load_LSC_model(path):
    model = tf.keras.models.load_model(
        path,
        custom_objects={
            'maLSNN': maLSNN, 'maLSNNb': maLSNNb, 'RateVoltageRegularization': RateVoltageRegularization,
            'AddLossLayer': AddLossLayer, 'AddMetricsLayer': AddMetricsLayer,
            'SparseCategoricalCrossentropy': tf.keras.losses.SparseCategoricalCrossentropy,
            'AdamW': AdamW2, 'DummyConstantSchedule': DummyConstantSchedule,
            'SymbolAndPositionEmbedding': SymbolAndPositionEmbedding,
            'ResLRUCell': ResLRUCell
        },
        compile=False
    )
    return model


def apply_LSC(train_task_args, model_args, batch_size, n_samples=-1, norm_pow=2, steps_per_epoch=2, es_epsilon=.08,
              patience=20, rec_norm=True, depth_norm=True, encoder_norm=True, decoder_norm=True, learn=True,
              time_steps=None, weights=None, save_weights_path=None, lr=1e-3, naswot=0, stop_time=None):
    time_string = timeStructured()
    print('LSC starts at: ', time_string)
    # FIXME: generalize this loop for any recurrent model
    gen_train = Task(**train_task_args)

    comments = model_args['comments']
    target_norm = str2val(comments, 'targetnorm', float, default=1)

    model_args['initial_state'] = ''

    ostack = model_args['stack']
    net_name = model_args['net_name']
    task_name = model_args['task_name']
    if isinstance(model_args['stack'], str):
        stack = [int(s) for s in model_args['stack'].split(':')]
    elif isinstance(model_args['stack'], int):
        stack = [model_args['n_neurons'] for _ in range(model_args['stack'])]
    else:
        stack = ostack

    s = model_args["seed"]
    weight_decay = 1e-3 if net_name == 'rsimplernn' else 1e-4
    # optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay)
    # adabelief = tfa.optimizers.AdaBelief()
    adabelief = tfa.optimizers.AdaBelief(lr=lr, weight_decay=weight_decay)
    optimizer = tfa.optimizers.Lookahead(adabelief, sync_period=6, slow_step_size=0.5)

    states = []

    losses = []
    all_norms = []
    all_naswot = []
    save_norms = {}

    li, pi, ni = None, None, None

    # the else is valid for the LSTM
    # hi, ci = (1, 2) if 'LSNN' in net_name else (0, 1)
    # n_states = 4 if 'LSNN' in net_name else 2

    ms = 1  # state size multiplier
    if 'LSNN' in net_name:
        if not 'reoldspike' in comments:
            hi, ci = 1, 2
            n_states = 3
        else:
            hi, ci = 0, 1
            n_states = 2

    elif 'LSTM' in net_name or 'lru' in net_name:
        hi, ci = 0, 1
        n_states = 2
        ms = 2

    else:
        hi, ci = 0, None
        n_states = 1

    for width in stack:
        for _ in range(n_states):
            states.append(tf.Variable(tf.zeros((batch_size, ms * width), dtype=tf.float32)))

    pbar1 = tqdm(total=steps_per_epoch, position=1)
    epsilon_steps = 0

    results = {}

    # get initial values of model
    model = build_model(**model_args)
    # model.summary()

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

    tape, norms = None, None

    pretrained_file = get_pretrained_file(comments, s, net_name, task_name, ostack)
    path_pretrained = os.path.join(GEXPERIMENTS, pretrained_file)

    if 'pretrained' in comments and 'findLSC' in comments:
        if os.path.exists(path_pretrained):
            print('Loading pretrained lsc weights')
            try:
                model = load_LSC_model(path_pretrained)
            except Exception as e:
                model = build_model(**model_args)
                print(e)

    if weights is None:
        weights = model.get_weights()
        best_weights = weights

    time_start = time.perf_counter()
    time_over = False
    randlsc = True if 'randlsc' in comments else False
    r1, r2, r3, r4 = 0, 0, 0, 0

    round_to = 4
    ma_loss, ma_norm = None, None

    best_norm = None
    best_loss = None
    best_individual_norms = {}
    n_types = ['rec', 'depth', 'enc', 'dec']
    best_count = 0
    failures = 0
    iterations = 0
    dec_norm = -1
    wnames = [weight.name for layer in model.layers for weight in layer.weights]

    std_ma_norm = 1
    if 'onlyloadpretrained' in comments:
        steps_per_epoch = 1
        time_steps = 2 if not 'test' in comments else time_steps
        learn = False
        std_ma_norm = 0

    if 'randlambda1' in comments:
        l = lambda: 2 * tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.float32)
    elif 'randlambda2' in comments:
        l = lambda: float(tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.float32) > .5)
    else:
        l = lambda: 1

    last_step = 0
    best_std_ma_norm = std_ma_norm
    stop_time = 60 * 60 * 16 if stop_time is None else stop_time - 30 * 60
    n_saves = 0
    std_thr = .5
    for step in range(steps_per_epoch):
        if time_over:
            break

        last_step = step
        for i, _ in enumerate(stack):
            for nt in n_types:
                save_norms[f'batch {step} {nt} layer {i}'] = []

        batch = gen_train.__getitem__(step)
        batch = [tf.Variable(tf.convert_to_tensor(tf.cast(b, tf.float32), dtype=tf.float32)) for b in batch[0]],

        if 'gausslsc' in comments:
            batch = [tf.random.normal(b.shape) for b in batch[0]],
            batch[0][1] = tf.convert_to_tensor(np.random.choice(gen_train.vocab_size, size=batch[0][1].shape))

        elif 'berlsc' in comments:
            batch = [tf.math.greater(tf.random.uniform(b.shape), .5) for b in batch[0]],
            batch = [tf.cast(b, dtype=tf.float32) for b in batch[0]],

        elif 'shufflelsc' in comments:
            shapes = [b.shape for b in batch[0]]
            flatbatch = [tf.reshape(b, [-1, 2]) for b in batch[0]],
            shuffled = [tf.random.shuffle(b) for b in flatbatch[0]],
            batch = [tf.reshape(b, s) for s, b in zip(shapes, shuffled[0])],

        elif 'randchlsc' in comments:
            shapes = [b.shape for b in batch[0]]
            batch = [tf.constant(np.random.choice(gen_train.vocab_size, size=s)) for s in shapes],

        batch[0][1] = tf.convert_to_tensor(np.random.choice(gen_train.vocab_size, size=batch[0][1].shape))

        ts = batch[0][0].shape[1] if time_steps is None else time_steps
        pbar2 = tqdm(total=ts, position=0)

        for t in range(ts):
            iterations += 1

            if True:
                # try:
                bt = tf.Variable(batch[0][0][:, t, :][:, None])
                wt = tf.Variable(batch[0][1][:, t][:, None])

                tf.keras.backend.clear_session()
                model, tape = None, None
                tf.keras.backend.clear_session()
                with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
                    tape.watch(wt)
                    tape.watch(bt)
                    tape.watch(states)

                    if not 'ptb' in task_name:
                        model = build_model(**model_args)
                    else:
                        model, noemb_model, embedding_model = build_model(**model_args, get_embedding=True, timesteps=1)

                    if not any([np.isnan(w.mean()) for w in weights]):
                        model.set_weights(weights)

                    if not 'ptb' in task_name:
                        bflat = tf.reshape(bt, [bt.shape[0], -1])
                        breshaped = tf.Variable(tf.reshape(bflat, bt.shape))

                        outputs = model([breshaped, wt, *states])
                    else:
                        bt = embedding_model(bt)

                        bflat = tf.reshape(bt, [bt.shape[0], -1])
                        breshaped = tf.Variable(tf.reshape(bflat, bt.shape))

                        outputs = noemb_model([breshaped, wt, *states])

                    states_p1 = outputs[1:]

                    mean_loss = 0
                    some_norms = []
                    some_losses = []
                    n_norms = len(stack) + 2 + len(stack) - 1
                    norms_names = []
                    state_below = None

                    for i, _ in enumerate(stack):

                        htp1 = states_p1[i * n_states + hi]
                        ht = states[i * n_states + hi]
                        stp1 = [htp1]
                        st = [ht]
                        ct, ctp1 = None, None
                        if not ci is None:
                            ctp1 = states_p1[i * n_states + ci]
                            ct = states[i * n_states + ci]
                            stp1.append(ctp1)
                            st.append(ct)

                        if randlsc:
                            r1, r2, r3, r4 = np.random.rand(4)

                        if rec_norm and r1 < .5:
                            rnorm, loss, naswot_score = get_norms(tape=tape, lower_states=st,
                                                                  upper_states=stp1,
                                                                  n_samples=n_samples, norm_pow=norm_pow, naswot=naswot,
                                                                  comments=comments, target_norm=target_norm)

                            save_norms[f'batch {step} rec layer {i}'].append(tf.reduce_mean(rnorm).numpy())
                            norms_names.append(f'rec l{i}')
                            some_norms.append(tf.reduce_mean(rnorm))
                            some_losses.append(loss)
                            if not naswot_score is None:
                                all_naswot.append(tf.reduce_mean(naswot_score))

                            mean_loss += l() * loss / n_norms

                        if encoder_norm and i == 0 and r2 < .5:
                            lower_states = [bflat]
                            norms, loss, naswot_score = get_norms(tape=tape, lower_states=lower_states,
                                                                  upper_states=stp1,
                                                                  n_samples=n_samples, norm_pow=norm_pow, naswot=naswot,
                                                                  comments=comments, target_norm=target_norm)
                            save_norms[f'batch {step} enc layer {i}'].append(tf.reduce_mean(norms).numpy())
                            norms_names.append(f'enc l{i}')

                            some_norms.append(tf.reduce_mean(norms))
                            some_losses.append(loss)
                            mean_loss += l() * loss / n_norms

                        sl = stp1

                        if depth_norm and r3 < .5:
                            if not state_below is None:
                                print('depth', '-' * 40)
                                norms, loss, naswot_score = get_norms(tape=tape, lower_states=state_below,
                                                                      upper_states=sl,
                                                                      n_samples=n_samples, norm_pow=norm_pow,
                                                                      naswot=naswot,
                                                                      comments=comments, target_norm=target_norm)
                                # print('here?!', norms)
                                save_norms[f'batch {step} depth layer {i}'].append(tf.reduce_mean(norms).numpy())
                                norms_names.append(f'depth l{i}')

                                some_norms.append(tf.reduce_mean(norms))
                                some_losses.append(loss)
                                mean_loss += l() * loss / n_norms

                        state_below = sl
                        del sl

                        if decoder_norm and i == len(stack) - 1 and r4 < .5:
                            output = outputs[0][:, 0, :]

                            if not output.shape[-1] > 100:
                                norms, loss, naswot_score = get_norms(tape=tape, lower_states=stp1,
                                                                      upper_states=[output],
                                                                      n_samples=n_samples, norm_pow=norm_pow,
                                                                      naswot=naswot,
                                                                      comments=comments, target_norm=1.)
                                save_norms[f'batch {step} dec layer {i}'].append(tf.reduce_mean(norms).numpy())
                                # norms_names.append(f'dec layer {i}')

                                # some_norms.append(tf.reduce_mean(norms))
                                some_losses.append(loss)
                                mean_loss += l() * loss / n_norms

                        del htp1, ht, ctp1, ct

                mean_norm = tf.reduce_mean(some_norms)
                ma_loss = loss if ma_loss is None else ma_loss * 9 / 10 + loss / 10
                ma_norm = mean_norm if ma_norm is None else ma_norm * 9 / 10 + mean_norm / 10
                std_ma_norm = std_ma_norm * 9 / 10 + np.std(some_norms) ** 2 / 10

                best_count += 1
                lower_than_target = True
                if not best_norm is None:
                    lower_than_target = mean_norm.numpy() < target_norm

                    if np.abs(mean_norm.numpy() - target_norm) < np.abs(best_norm - target_norm) \
                            and std_ma_norm < std_thr and learn:
                        best_norm = mean_norm.numpy()
                        best_loss = mean_loss.numpy()
                        best_std_ma_norm = std_ma_norm

                        for i, _ in enumerate(stack):
                            for nt in n_types:
                                # best_individual_norms[f'{nt} layer {i}'] = save_norms[f'batch {step} {nt} layer {i}'][-1]
                                a = save_norms[f'batch {step} {nt} layer {i}']
                                if not len(a) == 0:
                                    best_individual_norms[f'{nt} layer {i}'] = a[-1]
                                else:
                                    best_individual_norms[f'{nt} layer {i}'] = -1

                        best_weights = model.get_weights()
                        best_count = 0
                        n_saves += 1

                        if n_saves >= 2:
                            std_thr = .1

                        if 'pretrained' in comments and not model is None:
                            print('Saving pretrained lsc weights with best norms')
                            for i in range(len(model.weights)):
                                model.weights[i]._handle_name = model.weights[i].name + "_" + str(i)
                            model.save(path_pretrained)
                else:

                    for i, _ in enumerate(stack):
                        for nt in n_types:
                            a = save_norms[f'batch {step} {nt} layer {i}']
                            if not len(a) == 0:
                                best_individual_norms[f'{nt} layer {i}'] = a[-1]
                            else:
                                best_individual_norms[f'{nt} layer {i}'] = -1

                if learn and not 'nosgd' in comments:
                    grads = tape.gradient(mean_loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                    del grads

                states = states_p1

                print('\n')
                print(norms_names)
                print([n.numpy().round(3) for n in some_norms])

                if best_count > 2 * patience:
                    print('Reloading best weights')
                    weights = best_weights
                    best_count = 0

                if not np.isnan(mean_loss.numpy()):
                    del weights
                    weights = model.get_weights()

                    r = np.random.rand()
                    if 'wshuff' in comments and learn and r > .66:
                        new_weights = []
                        for w in weights:
                            if len(w.shape) >= 2:
                                oshape = w.shape
                                w = w.reshape(-1)
                                np.random.shuffle(w)
                                w = w.reshape(oshape)
                            new_weights.append(w)
                        weights = new_weights

                    if 'waddnoise' in comments and lower_than_target and learn:
                        print('adding noise to weights!')
                        new_weights = []
                        for w in weights:
                            if len(w.shape) >= 2:
                                noise = 1 * tf.random.uniform(w.shape, -1, 1) * tf.math.reduce_std(w)
                                w += noise.numpy()

                            new_weights.append(w)
                        weights = new_weights

                    if 'wmultiplier' in comments and learn:
                        print('multiplier to weights!')
                        new_weights = []
                        for w, _wname in zip(weights, wnames):
                            # print(_wname)
                            wname = _wname
                            if len(w.shape) >= 2 or 'tau' in wname or 'bias' in wname \
                                    or 'internal_current' in wname or '/thr:' in wname \
                                    or '/beta:' in wname or 'lambda' in wname:
                                n_multiplier, multiplier = 1, 1
                                wname = wname.replace('ma_lsnn_', '_cell_')
                                wname = wname.replace('ma_lsn_nb_', '_cell_')

                                if ('encoder_' in wname or '_cell_' in wname) \
                                        and not 'embedding' in wname:
                                    multiplier = target_norm / mean_norm.numpy()

                                    if 'encoder_' in wname:
                                        depth = int(wname.split('_')[1])
                                    else:
                                        depth = int(wname.split('cell_')[1].split('/')[0]) - 1

                                    dname = 'enc' if depth == 0 else 'depth'

                                    depth_radius = False
                                    if 'lsnn' in wname:
                                        if 'internal_current' in wname or 'thr:' in wname or 'beta:' in wname:
                                            depth_radius = True

                                    elif 'res_lru' in wname:
                                        if 'geglu' in wname and 'kernel' in wname:
                                            depth_radius = True

                                        if 'C_re' in wname or 'B_re' in wname or 'B_im' in wname or 'C_im' in wname:
                                            depth_radius = True

                                    elif 'lstm' in wname:
                                        if '/kernel:' in wname:
                                            depth_radius = True

                                    rec_radius = False
                                    if 'lsnn' in wname:
                                        if 'tau' in wname:
                                            rec_radius = True

                                    elif 'res_lru' in wname:
                                        if 'lambda_nu' in wname:
                                            rec_radius = True

                                    if 'input_weights' in wname or depth_radius:
                                        # or 'B_re' in wname or 'B_im' in wname:
                                        idx = norms_names.index(f'{dname} l{depth}')
                                        local_norm = some_norms[idx].numpy()
                                        n_multiplier = target_norm / local_norm
                                        # print('depth radius: ' + wname, n_multiplier)
                                        # print('   local norm: ', local_norm)

                                    elif 'recurrent_weights' in wname or 'recurrent_kernel' in wname or rec_radius:
                                        idx = norms_names.index(f'rec l{depth}')
                                        local_norm = some_norms[idx].numpy()
                                        n_multiplier = target_norm / local_norm
                                        # print('recurrent radius: ' + wname, n_multiplier)
                                        # print('   local norm: ', local_norm)

                                elif 'decoder/kernel' in wname or 'embedding' in wname:
                                    s = w.shape[1] if 'embedding' in wname else w.shape[0]
                                    w_norm = np.std(w) * np.sqrt(s)
                                    n_multiplier = 1 / w_norm
                                    dec_norm = w_norm

                                m = n_multiplier
                                m = np.clip(m, 0.85, 1.15)

                                w = m * w
                            new_weights.append(w)
                        weights = new_weights

                tf.keras.backend.clear_session()
                tf.keras.backend.clear_session()
                tf.keras.backend.clear_session()

                if time.perf_counter() - time_start > stop_time:  # 17h
                    time_over = True
                    break

                epsilons = [(abs(n - target_norm) < es_epsilon).numpy() for n in some_norms]
                if not ma_norm is None and all(epsilons) and std_ma_norm < std_thr:
                    epsilon_steps += 1
                else:
                    epsilon_steps = 0

                if epsilon_steps > patience:
                    time_over = True
                    break

                all_norms.append(mean_norm.numpy())
                losses.append(mean_loss.numpy())

                pbar2.update(1)

                prms = tf.reduce_mean([tf.reduce_mean(w) for w in model.trainable_weights]).numpy()
                if li is None and not mean_loss is None:
                    li = str(round(mean_loss.numpy(), 4))
                    pi = str(round(prms, 4))
                    ni = str(ma_norm.numpy().round(round_to))
                    best_norm = ma_norm.numpy()
                    best_loss = mean_loss.numpy()

                show_loss = str(ma_loss.numpy().round(round_to))
                show_norm = str(ma_norm.numpy().round(round_to))
                pbar2.set_description(
                    f"Step {step}; "
                    f"loss {str(show_loss)}/{li}; "
                    f"mean params {str(round(prms, round_to))}/{pi}; "
                    f"mean norms {show_norm}/{ni} (best {str(np.array(best_norm).round(round_to))}); "
                    f"ma std norms {str(np.array(std_ma_norm).round(round_to))}/{1} (best {str(np.array(best_std_ma_norm).round(round_to))}); "
                    f"fail rate {failures / iterations * 100:.1f}%; "
                )

            # except Exception as e:
            #     failures += 1
            #     print(e)

        del batch

        pbar1.update(1)
        if epsilon_steps > patience:
            break

    del gen_train

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

    all_norms.append(best_norm)
    losses.append(best_loss)

    tf.keras.backend.clear_session()
    tf.keras.backend.clear_session()
    tf.keras.backend.clear_session()

    if learn:
        for i, _ in enumerate(stack):
            for nt in n_types:
                save_norms[f'batch {last_step} {nt} layer {i}'].append(best_individual_norms[f'{nt} layer {i}'])

    results.update(
        LSC_losses=str(losses), LSC_norms=str(all_norms), save_norms=save_norms, all_naswot=str(all_naswot),
        fail_rate=failures / iterations
    )

    norms = save_norms
    keys = list(norms.keys())
    last_batch = np.unique([k[:8] for k in keys])[-1]
    print(last_batch)
    keys = [k for k in keys if last_batch in k]
    keys.sort()

    final_norms = []
    norm_names = []
    for k in keys:
        if not norms[k] == [-1] and not norms[k] == []:
            if not 'dec' in k:
                final_norms.append(norms[k][-1])
                norm_names.append(k.replace(' layer ', ' l'))
            elif dec_norm == -1:
                dec_norm = norms[k][-1]

    results['weights_shapes'] = [weight.shape for weight in best_weights]
    results['weights_names'] = wnames

    results['final_norms'] = final_norms
    results['norm_names'] = norm_names
    results['final_norm_dec'] = dec_norm
    results['final_norms_mean'] = np.mean(final_norms)
    results['final_norms_std'] = np.std(final_norms)
    results['std_ma_norm'] = std_ma_norm
    results['best_std_ma_norm'] = best_std_ma_norm

    print('Final norms:', final_norms)
    print('         ', norm_names)
    for name, norm in zip(norm_names, final_norms):
        print('      ', name, norm)

    print(f'     mean pm std: {np.mean(final_norms)} pm {np.std(final_norms)}')

    return best_weights, results
