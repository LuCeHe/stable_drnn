import tensorflow as tf
import time, os
import numpy as np
from tensorflow_addons.optimizers import AdamW
from tqdm import tqdm

from GenericTools.keras_tools.convenience_operations import sample_axis, desample_axis
from GenericTools.keras_tools.esoteric_losses import well_loss
from GenericTools.keras_tools.esoteric_tasks.numpy_generator import NumpyClassificationGenerator
from GenericTools.keras_tools.expose_latent import split_model, truer_split_model
from GenericTools.stay_organized.utils import flaggedtry
from alif_sg.neural_models.recLSC import get_norms, get_lsctype

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.abspath(os.path.join(CDIR, '..', 'good_experiments'))
os.makedirs(EXPERIMENTS, exist_ok=True)

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def get_weights_statistics(results, weight_names, weights):
    for n, w in zip(weight_names, weights):

        try:
            m = tf.reduce_mean(w).numpy()
        except Exception as e:
            m = None

        try:
            v = tf.math.reduce_variance(w).numpy()
        except Exception as e:
            v = None

        results[f'{n}_mean'].append(m)
        results[f'{n}_var'].append(v)

    return results


def apply_LSC_no_time(build_model, generator, max_dim=1024, n_samples=-1, norm_pow=2, forward_lsc=False,
                      nlayerjump=None, comments='', epsilon=.06, patience=20, learning_rate=3.16e-3,
                      subsample_axis=False,
                      skip_in_layers=[], skip_out_layers=[],
                      keep_in_layers=None, keep_out_layers=None,
                      net_name='', task_name='', seed=0,
                      activation='', custom_objects=None):
    assert callable(build_model)
    round_to = 4
    li, pi, ni = None, None, None

    # 1e1
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-4)

    all_norms = []
    all_losses = []

    results = {}

    # get initial values of model
    model = build_model()
    model.summary()

    weights = model.get_weights()
    weight_names = [weight.name for layer in model.layers for weight in layer.weights]
    results.update({f'{n}_mean': [] for n in weight_names})
    results.update({f'{n}_var': [] for n in weight_names})

    results = get_weights_statistics(results, weight_names, weights)

    lnames = [layer.name for layer in model.layers]

    if keep_in_layers is None:
        keep_in_layers = lnames
    if keep_out_layers is None:
        keep_out_layers = lnames
    inlnames = [
                   i for i, l in enumerate(lnames)
                   if not any([s in l for s in skip_in_layers])
                      and any([s in l for s in keep_in_layers])
               ][:-1]
    outlnames = [
                    i for i, l in enumerate(lnames)
                    if not any([s in l for s in skip_out_layers])
                       and any([s in l for s in keep_out_layers])
                ][1:]
    del model
    tf.keras.backend.clear_session()

    ma_loss, ma_norm, ma_factor = None, None, None
    show_loss, show_norm, show_avw, show_factor = None, None, None, None
    n_failures = 0
    loss, model = None, None

    lsct = get_lsctype(comments)
    path_pretrained = os.path.join(
        EXPERIMENTS, f"pretrained_s{seed}_{net_name}_{task_name}_{activation}_{lsct}.h5")
    if 'pretrained' in comments:
        if os.path.exists(path_pretrained):
            try:
                print('Loading pretrained lsc weights')
                model = tf.keras.models.load_model(path_pretrained, custom_objects=custom_objects)
                weights = model.get_weights()
            except Exception as e:
                model = None
                print(e)

    time_start = time.perf_counter()
    time_over = False

    for epoch in range(generator.epochs):
        pbar = tqdm(total=generator.steps_per_epoch)
        if time_over:
            break

        generator.on_epoch_end()
        for step in range(generator.steps_per_epoch):
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()

            if time.perf_counter() - time_start > 60 * 60 * 12:
                time_over = True
                break

            if not ma_norm is None and abs(ma_norm - 1.) < epsilon:
                epsilon_steps += 1
            else:
                epsilon_steps = 0

            if epsilon_steps > patience:
                break

            # if True:
            try:
                batch = generator.__getitem__(step)[0]
                if isinstance(batch, list) or isinstance(batch, tuple):
                    batch = [tf.convert_to_tensor(tf.cast(b, tf.float32), dtype=tf.float32) for b in batch]
                else:
                    batch = tf.convert_to_tensor(tf.cast(batch, tf.float32), dtype=tf.float32)
                tf.keras.backend.clear_session()

                with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
                    tape.watch(batch)

                    del model
                    tf.keras.backend.clear_session()

                    model = build_model()
                    if not weights is None:
                        model.set_weights(weights)

                    # print(lnames)
                    for _ in range(3):
                        inp_l = np.random.choice(inlnames)
                        outlist = [i for i in outlnames if i > inp_l]
                        out_l = np.random.choice(outlist)
                        pairs = [inp_l, out_l]
                        input_shape = model.layers[pairs[0] + 1].input_shape[1:]
                        if not isinstance(input_shape, list):
                            break

                    if 'alllays' in comments:
                        init_pairs = list(range(len(lnames)))
                    else:
                        init_pairs = [pairs[0]]

                    loss = 0
                    for ip in init_pairs:
                        # print(ip,len(init_pairs))
                        if isinstance(nlayerjump, int):
                            pairs[1] = ip + nlayerjump

                        # print('\n\n')
                        # print(pairs, len(lnames))
                        # print(lnames[pairs[0]], lnames[pairs[1]])
                        if not 'truersplit' in comments:
                            premodel, intermodel = split_model(model, pairs)
                        else:
                            premodel, intermodel = truer_split_model(model, pairs)

                        preinter = premodel(batch)
                        del premodel
                        allpreinter = preinter

                        if isinstance(allpreinter, list):
                            preinter = allpreinter[0]
                        tape.watch(preinter)

                        if subsample_axis and not forward_lsc:
                            # flatten and deflatten to make sure the flat version is in the gradient graph
                            # preinter_shape = preinter.shape
                            flat_inp = tf.reshape(preinter, [preinter.shape[0], -1])

                            # sample and desample to make sure that the sample is in the graph,
                            # so the derivative will be taken correctly
                            shuffinp, reminder, indices = sample_axis(flat_inp, max_dim=max_dim,
                                                                      return_deshuffling=True)
                            defhuffledinp = desample_axis(shuffinp, reminder, indices)

                            assert tf.math.reduce_all(tf.equal(defhuffledinp, flat_inp))

                            new_preinter = tf.reshape(defhuffledinp, preinter.shape)

                            assert tf.math.reduce_all(tf.equal(new_preinter, preinter))

                        elif 'deslice' in comments:
                            shape = preinter.shape
                            ones = np.array(shape) == 1
                            deslice_axis = list(range(len(shape)))
                            deslice_axis = [a for a, b in zip(deslice_axis, ones) if b == False and not a == 0]

                            np.random.shuffle(deslice_axis)
                            deslice_axis = deslice_axis[:-1]

                            st = preinter
                            reminders = []
                            deshuffles = []
                            for axis in deslice_axis:
                                st, remainder, deshuffle_indices = sample_axis(st, max_dim=1, return_deshuffling=True,
                                                                               axis=axis)
                                reminders.append(remainder)
                                deshuffles.append(deshuffle_indices)

                            # squeeze
                            oshape = st.shape
                            unsq_idx = [i for i, s in enumerate(oshape) if s == 1]
                            st = tf.squeeze(st)

                            slice = st

                            # unsqueeze
                            rt = slice
                            for i in unsq_idx:
                                rt = tf.expand_dims(rt, i)

                            st = rt

                            for j, _ in enumerate(deslice_axis):
                                i = -j - 1
                                st = desample_axis(st, reminders[i], deshuffles[i], axis=deslice_axis[i])

                            new_preinter = st

                        else:
                            new_preinter = preinter

                        if isinstance(allpreinter, list):
                            allpreinter[0] = new_preinter
                        else:
                            allpreinter = new_preinter

                        interout = intermodel(allpreinter)

                        del allpreinter

                        if not forward_lsc:
                            if isinstance(interout, list) or isinstance(interout, tuple):
                                idx = np.random.choice(len(interout))
                                interout = interout[idx]
                            oup = interout

                            if subsample_axis:
                                inp = shuffinp

                            elif 'deslice' in comments:
                                inp = slice

                                shape = interout.shape
                                ones = np.array(shape) == 1
                                deslice_axis = list(range(len(shape)))
                                deslice_axis = [a for a, b in zip(deslice_axis, ones) if b == False and not a == 0]

                                np.random.shuffle(deslice_axis)
                                deslice_axis = deslice_axis[:-1]

                                st = interout
                                reminders = []
                                deshuffles = []
                                for axis in deslice_axis:
                                    st, remainder, deshuffle_indices = sample_axis(st, max_dim=1,
                                                                                   return_deshuffling=True,
                                                                                   axis=axis)
                                    reminders.append(remainder)
                                    deshuffles.append(deshuffle_indices)
                                oup = tf.squeeze(st)

                            else:
                                inp = preinter

                            reoup = tf.reshape(oup, [oup.shape[0], -1])
                            oup = sample_axis(reoup, max_dim=max_dim)

                            norms, iloss, naswot_score = get_norms(tape, [inp], [oup], n_samples=n_samples,
                                                                   norm_pow=norm_pow, comments=comments)
                            if (norms.numpy() == 1).all():
                                raise ValueError('Norms are all 1, since the input and output are the same. '
                                                 'This happens because the architecture is complex and now '
                                                 'we are not able to find a path from the input to the output '
                                                 'in the general case.')

                        else:
                            varin = tf.math.reduce_variance(new_preinter)
                            varout = tf.math.reduce_variance(interout)
                            iloss = tf.reduce_mean(tf.abs(varin - varout))
                            norms = varout / varin
                        loss += iloss

                    ma_loss = loss if ma_loss is None else ma_loss * 9 / 10 + loss / 10
                    norm = tf.reduce_mean(norms)
                    ma_norm = norm if ma_norm is None else ma_norm * 9 / 10 + norm / 10
                    all_norms.append(norm.numpy())
                    all_losses.append(loss.numpy())

                grads = tape.gradient(loss, intermodel.trainable_weights)
                optimizer.apply_gradients(zip(grads, intermodel.trainable_weights))
                del intermodel
                tf.keras.backend.clear_session()
                tf.keras.backend.clear_session()

                new_weights = model.get_weights()
                av_weights = tf.reduce_mean([tf.reduce_mean(tf.cast(t, tf.float32)) for t in new_weights])
                if not tf.math.is_nan(av_weights):
                    weights = new_weights

                results = get_weights_statistics(results, weight_names, weights)

                # show_factor = str(np.array(ma_factor).round(3))
                show_loss = str(ma_loss.numpy().round(round_to))
                show_norm = str(ma_norm.numpy().round(round_to))
                show_avw = str(av_weights.numpy().round(round_to))

            except Exception as e:
                print(e)
                n_failures += 1

            if li is None:
                li = show_loss
                pi = show_avw
                ni = show_norm

            show_failure = str(np.array(n_failures / ((step + 1) + epoch * generator.steps_per_epoch)).round(round_to))
            pbar.update(1)
            pbar.set_description(
                f"Pretrain e {epoch + 1} s {step + 1}, "
                f"Loss {show_loss}/{li}, Norms {show_norm}/{ni}, "
                # f"Factor {show_factor}, "
                f"Av. Weights {show_avw}/{pi}, "
                # f"Failures {n_failures}, "
                f"Fail rate {show_failure}, "
                f"ES {epsilon_steps}/{patience} "

            )

        if epsilon_steps > patience:
            break

    fail_rate = n_failures / generator.epochs / generator.steps_per_epoch

    if 'pretrained' in comments and not model is None:
        if (float(show_norm) - 1) < (float(ni) - 1):
            model.save(path_pretrained)

    model, generator = None, None
    del model, generator

    for n in weight_names:
        results[f'{n}_mean'] = str(results[f'{n}_mean'])
        results[f'{n}_var'] = str(results[f'{n}_var'])

    results.update(LSC_losses=str(all_losses), LSC_norms=str(all_norms), LSC_fail_rate=str(fail_rate))
    tf.keras.backend.clear_session()
    tf.keras.backend.clear_session()
    tf.keras.backend.clear_session()
    return weights, results
