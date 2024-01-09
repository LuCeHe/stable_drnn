import tensorflow as tf
import numpy as np
from pyaromatics.keras_tools.convenience_operations import sample_axis
from pyaromatics.keras_tools.esoteric_losses import well_loss

def get_norms(tape=None, lower_states=None, upper_states=None, n_samples=-1, norm_pow=2, naswot=0, comments='',
              log_epsilon=1e-8, target_norm=1., n_s=16, test=False):
    if tape is None and lower_states is None and upper_states is None and test == False:
        raise ValueError('No input data given!')

    norms = None
    loss = 0
    # upper_states = [tf.squeeze(hl) for hl in upper_states]
    if not test:
        hss = []
        for hlm1 in lower_states:
            hs = [tape.batch_jacobian(hl, hlm1, experimental_use_pfor=True) for hl in upper_states]
            # hs_aux = [tape.batch_jacobian(hlm1,hl,  experimental_use_pfor=True) for hl in upper_states]
            # print(tape.gradient(upper_states[0], hlm1))
            # print(tape.gradient(hlm1, upper_states[0]))
            # print(hs)
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
        # norms = tf.math.reduce_max(tf.abs(tf.linalg.eigvals(std)), axis=-1)
        eigs, _ = tf.linalg.eig(std)
        norms = tf.math.reduce_max(tf.abs(eigs), axis=-1)

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
