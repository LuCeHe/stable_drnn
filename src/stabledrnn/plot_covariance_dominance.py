import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stable_spike', 'src'))

from itertools import combinations
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import pandas as pd

EXPSDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'experiments')
table_path = os.path.join(EXPSDIR, 'covariance_dominance.csv')

get_radiuses = True
batch_size = 32
units = 16
time_steps = 100
max_order = 4
net_name = 'alif'  # gru lstm alif
seeds = 6  # 6
ys = None
nets = ['lstm', 'gru', 'alif']
datasets = ['mnist', 'cifar10']

# datasets = ['cifar10']
# nets = ['lstm']

if not os.path.exists(table_path):

    rows = []
    i = 0
    for dataset in datasets:
        if dataset == 'mnist':
            # Load MNIST dataset
            (x_train, _), _ = mnist.load_data()
        elif dataset == 'cifar10':
            # Load CIFAR-10 dataset
            (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
            # mean across all channels
            x_train = np.mean(x_train, axis=-1)
        else:
            raise ValueError('Unknown dataset')

        # Normalize input images
        print('x_train.shape', x_train.shape)
        x_train = x_train[..., None] / 255.0
        # resize from 28x28 to 10x10
        x_train = tf.image.resize(x_train, [int(np.sqrt(time_steps))] * 2).numpy()
        print('x_train.shape', x_train.shape)

        # reshape input images
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2], 1)

        dense = tf.keras.layers.Dense(units, activation='softmax')
        x_train = dense(x_train)
        batch = tf.convert_to_tensor(x_train[:batch_size, :time_steps], dtype=tf.float32)

        print('x_train.shape', x_train.shape)

        for net_name in nets:
            for seed in range(seeds):
                i += 1
                print(f'{i}/{len(nets) * seeds * len(datasets)}: net {net_name}, seed {seed}, dataset {dataset}')

                if get_radiuses:

                    if net_name == 'lstm':
                        model = tf.keras.layers.LSTM(units, input_shape=(None, units), activation='relu',
                                                     return_sequences=True)
                    elif net_name == 'gru':
                        model = tf.keras.layers.GRU(units, input_shape=(None, units), activation='relu',
                                                    return_sequences=True)
                    elif net_name == 'alif':
                        import stable_spike.src.stablespike.neural_models as models

                        comments = ''
                        cell = models.net('maLSNN')(num_neurons=units, config=comments)
                        model = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True, stateful=True)
                    elif net_name == 'lru':
                        from lru_unofficial.src.lruun.tf.linear_recurrent_unit import ResLRUCell

                        cell = ResLRUCell(num_neurons=units)
                        model = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True)
                    else:
                        raise ValueError('Unknown net_name')

                    # Use GradientTape to compute gradients
                    with tf.GradientTape(persistent=True) as tape:
                        tape.watch(batch)
                        output = model(batch)

                    if isinstance(output, list):
                        output = output[0]

                    # Compute gradients
                    jacobian = tape.batch_jacobian(output, batch)

                    # move axis 2 to -1 if jacobian.shape = (2, 4, 5, 4, 5) -> (2, 4, 4, 5, 5)
                    jacobian = tf.transpose(jacobian, perm=[0, 1, 3, 2, 4])

                    eigs = tf.abs(tf.linalg.eigvals(jacobian, name=None))
                    radius = tf.math.reduce_max(eigs, axis=-1)

                    # radius_l is the diagonal
                    # radius_l = tf.linalg.diag_part(radius)

                    # radius_t is the upper triangular part
                    radius = radius[:, 1:, :-1]
                    radius_t = tf.linalg.diag_part(radius)

                    ys = radius_t - tf.reduce_mean(radius_t, axis=0)
                    ys = ys.numpy()


                def get_interaction(ys, order=3):
                    assert len(ys.shape) == 2
                    time_steps = ys.shape[1]

                    a = list(range(time_steps))

                    combination = [sorted(list(c)) for c in list(combinations(a, order))]
                    all_exyz = np.mean(np.prod(ys[:, combination], axis=2), axis=0)
                    interaction = abs(np.sum(all_exyz))

                    return interaction


                if ys is None:
                    ys = np.random.rand(batch_size, time_steps)

                for order in range(2, max_order + 1):
                    interaction = get_interaction(ys, order=order)
                    print(f'interaction of order {order}: {interaction}')
                    row = {'net': net_name, 'seed': seed, 'order': order, 'interaction': interaction,
                           'dataset': dataset}

                    rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(table_path, index=False)

else:
    df = pd.read_csv(table_path)

# capitalize net column
df['net'] = df['net'].str.upper()

print(df.to_string())

# compute mean and std of interaction, with groupby net and order
stats_oi = ['mean', 'std']
mdf = df.groupby(['net', 'order', 'dataset']).agg({'interaction': stats_oi})

m = 'interaction'
for s in stats_oi:
    mdf[f'{s}_{m}'] = mdf[m][s]
mdf = mdf.drop([m], axis=1)
mdf = mdf.droplevel(level=1, axis=1)

print(mdf.to_string())

# plot a bar plot with error bars
import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols=len(datasets), figsize=(8, 3))

for i, dataset in enumerate(datasets):
    ax = axs[i]
    mdf.xs(dataset, level='dataset').unstack().plot(
        kind='bar', y='mean_interaction', yerr='std_interaction', ax=ax,
        color=['#2F5E0F', '#4C951A', '#6ACA28'],
    )

    if i == 0:
        ax.set_ylabel('Interaction Magnitude')
        ax.set_xlabel('Layer')
    else:
        ax.set_ylabel('')
        ax.set_xlabel('')

    if not i == len(datasets) - 1:
        ax.get_legend().remove()
    else:
        # higher
        # ax.legend(title='order', loc='upper right')
        # lower
        ax.legend(title='order', loc='upper left')

    ax.set_title(f'{dataset.upper()}')

    ax.set_yscale('log')
    # rotate xticks
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # remove axs lines
    for spine in ax.spines.values():
        spine.set_visible(False)

# save
fig.savefig(os.path.join(EXPSDIR, 'covariance_dominance.png'), bbox_inches='tight')
plt.show()
