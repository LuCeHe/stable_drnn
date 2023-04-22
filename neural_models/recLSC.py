import os, time, shutil
import numpy as np
import warnings
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_addons.optimizers import AdamW
from GenericTools.keras_tools.esoteric_optimizers.AdamW import AdamW as AdamW2

from GenericTools.keras_tools.convenience_operations import sample_axis
from GenericTools.keras_tools.esoteric_layers import AddLossLayer, AddMetricsLayer, SymbolAndPositionEmbedding
from GenericTools.keras_tools.esoteric_layers.rate_voltage_reg import RateVoltageRegularization
from GenericTools.keras_tools.learning_rate_schedules import DummyConstantSchedule
from GenericTools.stay_organized.utils import str2val, timeStructured
from sg_design_lif.neural_models import maLSNN, maLSNNb
from sg_design_lif.neural_models.config import default_config

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

        else:
            raise NotImplementedError

    elif n_samples > 0:
        x = tf.random.normal((td.shape[0], td.shape[-1], n_samples))
        x_norm = tf.norm(x, ord=norm_pow, axis=1)
        e = tf.einsum('bij,bjk->bik', td, x)
        e_norm = tf.norm(e, ord=norm_pow, axis=1)

        norms = e_norm / x_norm
        norms = tf.reduce_max(norms, axis=-1)

    else:
        norms = tf.math.reduce_max(tf.abs(tf.linalg.eigvals(std)), axis=-1)

    loss += well_loss(min_value=target_norm, max_value=target_norm, walls_type='relu', axis='all')(norms)

    naswot_score = None
    if not naswot == 0:
        batch_size = td.shape[0]
        t = tf.reshape(td, (batch_size, -1))

        if not 'v2naswot' in comments:
            t = sample_axis(t, max_dim=batch_size)
            std = tf.math.reduce_std(t)
            t = t + tf.random.normal(t.shape) * std / 10
            t = tf.tanh(t)

        else:
            std = tf.math.reduce_std(t)
            t = t + tf.random.normal(t.shape) * std / 10

            t = tf.transpose(t, (1, 0))
            t = tfp.stats.correlation(t)

        naswot_score = tf.abs(tf.linalg.slogdet(t)[1])

        naswot_loss = well_loss(min_value=0, max_value=0, walls_type='relu', axis='all')(naswot_score)
        naswot_loss = naswot_loss if not tf.math.is_inf(naswot_loss) else tf.ones_like(naswot_loss)

        if naswot == 1:
            scale_factor = tf.stop_gradient(loss / naswot_loss)
            loss += scale_factor * naswot_loss
        elif naswot == -1:
            scale_factor = 1  # tf.stop_gradient(1 / naswot_loss)
            loss = scale_factor * naswot_loss

    return tf.abs(norms), loss, naswot_score


def get_lsctype(comments):
    if 'supnpsd' in comments:
        lsctype = 'supnpsd'

    elif 'supsubnpsd' in comments:
        lsctype = 'supsubnpsd'

    elif 'logradius' in comments:
        lsctype = 'logradius'

    elif 'radius' in comments:
        lsctype = 'radius'

    elif 'entrywise' in comments:
        lsctype = 'entrywise'

    else:
        lsctype = 'other'
    return lsctype


def get_pretrained_file(comments, s, net_name, task_name, ostack):
    target_norm = str2val(comments, 'targetnorm', float, default=1)
    if ostack == 'None':
        ostack = None
    elif ostack in ['1', '3', '5', '7']:
        ostack = int(ostack)

    stack, batch_size, embedding, n_neurons, lr = default_config(
        ostack, None, None, None, .1, task_name, net_name, setting='LSC'
    )

    c = ''
    if 'targetnorm' in comments:
        c += f'_tn{str(target_norm).replace(".", "p")}'

    if 'randlsc' in comments:
        c += '_randlsc'

    if 'lscshuffw' in comments:
        c += '_lscshuffw'

    if 'gausslsc' in comments:
        c += '_gausslsc'

    if 'learnsharp' in comments:
        c += '_ls'

    if 'learndamp' in comments:
        c += '_ld'

    lsct = get_lsctype(comments)
    return f"pretrained_s{s}_{net_name}_{lsct}_{task_name}_stack{str(stack).replace(':', 'c')}{c}.h5"


def remove_pretrained_extra(experiments, remove_opposite=True, folder=None):
    files = []
    print('Desired:')
    for exp in experiments:
        file = get_pretrained_file(
            comments=exp['comments'][0],
            s=exp['seed'][0],
            net_name=exp['net'][0],
            task_name=exp['task'][0],
            ostack=exp['stack'][0]
        )
        print(file)
        files.append(file)

    if folder is None:
        folder = GEXPERIMENTS

    safety_folder = os.path.abspath(os.path.join(folder, '..', 'safety'))
    os.makedirs(safety_folder, exist_ok=True)

    existing_pretrained = [d for d in os.listdir(folder) if 'pretrained_' in d and '.h5' in d]
    pbar = tqdm(total=len(existing_pretrained))
    removed = 0
    for d in existing_pretrained:
        # copy d file to safety folder
        shutil.copy(os.path.join(folder, d), os.path.join(safety_folder, d))

        if not d in files and remove_opposite:
            os.remove(os.path.join(folder, d))
            removed += 1

        if d in files and not remove_opposite:
            os.remove(os.path.join(folder, d))
            removed += 1

        pbar.update(1)
        pbar.set_description(f"Removed {removed} of {len(existing_pretrained)}")

    which_is_missing = [f for f in files if not f in existing_pretrained]
    print('Missing:')
    for f in which_is_missing:
        print(f)


def load_LSC_model(path):
    model = tf.keras.models.load_model(
        path,
        custom_objects={
            'maLSNN': maLSNN, 'maLSNNb': maLSNNb, 'RateVoltageRegularization': RateVoltageRegularization,
            'AddLossLayer': AddLossLayer, 'AddMetricsLayer': AddMetricsLayer,
            'SparseCategoricalCrossentropy': tf.keras.losses.SparseCategoricalCrossentropy,
            'AdamW': AdamW2, 'DummyConstantSchedule': DummyConstantSchedule,
            'SymbolAndPositionEmbedding': SymbolAndPositionEmbedding,

        }
    )
    return model


def apply_LSC(train_task_args, model_args, norm_pow, n_samples, batch_size, steps_per_epoch=2, es_epsilon=.08,
              patience=5, rec_norm=True, depth_norm=True, encoder_norm=False, decoder_norm=True, learn=True,
              time_steps=None, weights=None, save_weights_path=None, lr=1e-3, naswot=0):
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
    weight_decay = 1e-3 if net_name == 'rsimplernn' and task_name == 'wordptb' else 1e-4
    optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay)

    states = []

    losses = []
    all_norms = []
    all_naswot = []
    save_norms = {}

    li, pi, ni = None, None, None

    # the else is valid for the LSTM
    # hi, ci = (1, 2) if 'LSNN' in net_name else (0, 1)
    # n_states = 4 if 'LSNN' in net_name else 2

    if 'LSNN' in net_name:
        hi, ci = 1, 2
        n_states = 4

    elif 'LSTM' in net_name:
        hi, ci = 0, 1
        n_states = 2

    else:
        hi, ci = 0, None
        n_states = 1

    for width in stack:
        for _ in range(n_states):
            states.append(tf.zeros((batch_size, width)))

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

    if 'onlyloadpretrained' in comments:
        steps_per_epoch = 1
        time_steps = 10 if not 'test' in comments else 2
        learn = False

    if 'randlambda' in comments:
        l = lambda: tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.float32)
    else:
        l = lambda: 1

    last_step = 0
    for step in range(steps_per_epoch):
        if time_over:
            break

        last_step = step
        for i, _ in enumerate(stack):
            for nt in n_types:
                save_norms[f'batch {step} {nt} layer {i}'] = []

        batch = gen_train.__getitem__(step)
        batch = [tf.convert_to_tensor(tf.cast(b, tf.float32), dtype=tf.float32) for b in batch[0]],

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

            # if True:
            try:
                bt = batch[0][0][:, t, :][:, None]
                wt = batch[0][1][:, t][:, None]

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
                        breshaped = tf.reshape(bflat, bt.shape)

                        outputs = model([breshaped, wt, *states])
                    else:
                        bt = embedding_model(bt)

                        bflat = tf.reshape(bt, [bt.shape[0], -1])
                        breshaped = tf.reshape(bflat, bt.shape)

                        outputs = noemb_model([breshaped, wt, *states])

                    states_p1 = outputs[1:]

                    mean_loss = 0
                    some_norms = []
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
                            some_norms.append(tf.reduce_mean(rnorm))
                            if not naswot_score is None:
                                all_naswot.append(tf.reduce_mean(naswot_score))
                            mean_loss += l() * loss

                        if encoder_norm and i == 0 and r2 < .5:
                            lower_states = [bflat]
                            norms, loss, naswot_score = get_norms(tape=tape, lower_states=lower_states,
                                                                  upper_states=stp1,
                                                                  n_samples=n_samples, norm_pow=norm_pow, naswot=naswot,
                                                                  comments=comments, target_norm=target_norm)
                            save_norms[f'batch {step} enc layer {i}'].append(tf.reduce_mean(norms).numpy())

                            some_norms.append(tf.reduce_mean(norms))
                            mean_loss += l() * loss

                        sl = stp1
                        # hl = htp1
                        # cl = ctp1
                        if depth_norm and r3 < .5:
                            if not state_below is None:
                                # hlm1, clm1 = state_below
                                norms, loss, naswot_score = get_norms(tape=tape, lower_states=state_below,
                                                                      upper_states=sl,
                                                                      n_samples=n_samples, norm_pow=norm_pow,
                                                                      naswot=naswot,
                                                                      comments=comments, target_norm=target_norm)
                                save_norms[f'batch {step} depth layer {i}'].append(tf.reduce_mean(norms).numpy())

                                some_norms.append(tf.reduce_mean(norms))
                                mean_loss += l() * loss

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

                                # some_norms.append(tf.reduce_mean(norms))
                                mean_loss += l() * loss

                        del htp1, ht, ctp1, ct

                best_count += 1
                mean_norm = tf.reduce_mean(some_norms)
                if not best_norm is None:
                    if np.abs(mean_norm.numpy() - target_norm) < np.abs(best_norm - target_norm):
                        best_norm = mean_norm.numpy()
                        best_loss = mean_loss.numpy()

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

                        if 'pretrained' in comments and not model is None and learn:
                            print('Saving pretrained lsc weights with best norms')
                            os.remove(path_pretrained)
                            model.save(path_pretrained)
                else:

                    for i, _ in enumerate(stack):
                        for nt in n_types:
                            a = save_norms[f'batch {step} {nt} layer {i}']
                            if not len(a) == 0:
                                best_individual_norms[f'{nt} layer {i}'] = a[-1]
                            else:
                                best_individual_norms[f'{nt} layer {i}'] = -1

                if learn:
                    grads = tape.gradient(mean_loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                    del grads

                states = states_p1

                if best_count > 2 * patience:
                    print('Reloading best weights')
                    weights = best_weights
                    best_count = 0

                if not np.isnan(mean_loss.numpy()):
                    del weights
                    weights = model.get_weights()
                    if 'lscshuffw' in comments:
                        for w in weights:
                            oshape = w.shape
                            w = w.reshape(-1)
                            np.random.shuffle(w)
                            w = w.reshape(oshape)

                tf.keras.backend.clear_session()
                tf.keras.backend.clear_session()
                tf.keras.backend.clear_session()

                if time.perf_counter() - time_start > 60 * 60 * 17:  # 17h
                    time_over = True
                    break

                ma_loss = loss if ma_loss is None else ma_loss * 9 / 10 + loss / 10
                ma_norm = mean_norm if ma_norm is None else ma_norm * 9 / 10 + mean_norm / 10

                epsilons = [(abs(n - target_norm) < es_epsilon).numpy() for n in some_norms]
                if not ma_norm is None and all(epsilons):
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
                    f"fail rate {failures / iterations * 100:.1f}%; "
                )

            except Exception as e:
                failures += 1
                print(e)

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

    if os.path.exists(path_pretrained):
        print('Loading pretrained lsc weights')
        try:
            model = load_LSC_model(path_pretrained)
            weights = model.get_weights()

        except Exception as e:
            print(e)

    for i, _ in enumerate(stack):
        for nt in n_types:
            save_norms[f'batch {last_step} {nt} layer {i}'].append(best_individual_norms[f'{nt} layer {i}'])

    results.update(
        LSC_losses=str(losses), LSC_norms=str(all_norms), save_norms=save_norms, all_naswot=str(all_naswot),
        fail_rate=failures / iterations
    )

    return weights, results


def test_1():
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


def test_slogdet():
    batch_size = 7
    t = tf.random.normal((batch_size, 10, 10))
    t = tf.reshape(t, (batch_size, -1))
    shuffinp = sample_axis(t, max_dim=batch_size)
    t = tf.tanh(shuffinp)
    naswot_score = tf.linalg.slogdet(t)[1]

    print(t.shape, shuffinp.shape)


def test_subsample_larger_axis():
    batch_size = 7
    t = tf.random.normal((batch_size, 12, 13))
    if tf.math.greater(t.shape[1], t.shape[2]):
        sample_ax = 1
        max_dim = t.shape[2]
    else:
        sample_ax = 2
        max_dim = t.shape[1]

    shuffinp = sample_axis(t, max_dim=max_dim, axis=sample_ax)
    print(shuffinp.shape)


def test_subsampled_larger_axis():
    batch_size = 7
    comments = '_supsubnpsd'
    tape = tf.random.normal((batch_size, 12, 13))
    get_norms(tape, lower_states=[], upper_states=[], n_samples=-1, norm_pow=2, naswot=0, comments=comments,
              epsilon=1e-8,
              target_norm=1., test=True)


def test_remove():
    experiments = [{'comments': [
        'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_pretrained_findLSC_radius_targetnorm:.5_onlypretrain'],
        'seed': [0], 'stack': ['None'], 'net': ['maLSNN'], 'task': ['heidelberg']}, {'comments': [
        'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_pretrained_findLSC_radius_onlypretrain'], 'seed': [1],
        'stack': ['None'],
        'net': ['maLSNN'],
        'task': [
            'heidelberg']}, {
        'comments': [
            'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_pretrained_findLSC_radius_targetnorm:.5_onlypretrain'],
        'seed': [1], 'stack': ['None'], 'net': ['maLSNN'], 'task': ['heidelberg']}, {'comments': [
        'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_pretrained_findLSC_radius_onlypretrain'], 'seed': [2],
        'stack': ['None'],
        'net': ['maLSNN'],
        'task': [
            'heidelberg']},
        {'comments': [
            'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_pretrained_findLSC_radius_onlypretrain'],
            'seed': [41], 'stack': ['4:3'], 'net': ['LSTM'], 'task': ['heidelberg']},
        {'comments': [
            'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_pretrained_findLSC_radius_onlypretrain'],
            'seed': [41],
            'stack': ['None'],
            'net': ['maLSNN'],
            'task': [
                'heidelberg']}]

    remove_pretrained_extra(experiments)


if __name__ == '__main__':
    test_remove()
