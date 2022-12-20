import time

print(1)
# measure time
start = time.time()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stay_organized.pandardize import experiments_to_pandas
from stay_organized.standardize_strings import shorten_losses

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
GEXPERIMENTS = [
    os.path.join(CDIR, 'good_experiments'),
    r'D:\work\alif_sg\good_experiments\2022-11-23--unclear_rnn_good_ffn'
]

plot_norms_evol = False
plot_norms_evol_1 = False
lrs_plot = False

print(2, time.time() - start)
start = time.time()

metric = 'val_acc max'  # 'val_acc max'   'val_loss min'
expsid = 'ffnandcnns'  # effnet als ffnandcnns
h5path = os.path.join(EXPERIMENTS, f'summary_{expsid}.h5')

df = experiments_to_pandas(
    h5path=h5path, zips_folder=GEXPERIMENTS, unzips_folder=EXPERIMENTS, experiments_identifier=expsid,
    exclude_files=['cout.txt'], check_for_new=True
)

print(list(df.columns))

print(3, time.time() - start)

start = time.time()

# select only rows with width 10
df['time_elapsed'] = pd.to_timedelta(df['time_elapsed'], unit='s')

df = df[df['width'] == 128]
df = df[df['layers'] == 30]
# df = df[df['comments'].str.contains('findLSC_supn')]
df = df[~df['comments'].str.contains('findLSC_radius_supn')]

if plot_norms_evol:
    for _, row in df.iterrows():

        fig, axs = plt.subplots(2, 3)
        fig.suptitle(row['comments'])

        for k in row.keys():
            try:
                if '_mean' in k:
                    curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                    axs[0, 0].plot(curve)
                    axs[0, 0].title.set_text('mean')

                if '_var' in k:
                    curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                    axs[0, 1].plot(curve)
                    axs[0, 1].title.set_text('variance')

                if 'LSC_losses' in k:
                    curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                    axs[1, 0].plot(curve)
                    axs[1, 0].title.set_text('lsc loss')

                if 'LSC_norms' in k:
                    curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                    axs[1, 1].plot(curve)
                    axs[1, 1].title.set_text('lsc norm')

                if 'sparse_categorical_accuracy list' == k:
                    axs[1, 2].plot(row[k])
                    axs[1, 2].title.set_text('acc')

                if 'val_sparse_categorical_accuracy list' == k:
                    axs[1, 2].plot(row[k], '--')
                    axs[1, 2].title.set_text('acc')

                if 'loss list' == k:
                    axs[0, 2].plot(row[k])
                    axs[0, 2].title.set_text('loss')

                if 'val_loss list' == k:
                    axs[0, 2].plot(row[k], '--')
                    axs[0, 2].title.set_text('loss')

            except Exception as e:
                print(e)

        plt.show()

if plot_norms_evol_1:
    df = df[df['width'] == 128]
    df = df[df['layers'] == 30]

    print(df.shape)
    fig, axs = plt.subplots(df.shape[0], 6)

    for j, row in df.iterrows():
        i = j - 1

        axs[i, 0].set_ylabel(row['comments'])
        print('-' * 20)
        kernel_names = sorted(np.unique([k.replace('_mean', '') for k in row.keys() if 'kernel:0_mean' in k]))
        print(kernel_names)
        print(len(kernel_names))
        for k in row.keys():
            if not 'bias' in k:
                try:
                    if '_mean' in k:
                        print(k)
                        # print(row['comments'],row['path'], k)
                        if isinstance(row[k], str):
                            print(row[k][:10])
                        else:
                            print(row[k])
                        curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                        axs[i, 0].plot(curve)
                        axs[i, 0].title.set_text('mean')

                    if '_var' in k:
                        # print(row[k])
                        curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                        axs[i, 1].plot(curve)
                        axs[i, 1].title.set_text('variance')

                    if 'LSC_losses' in k:
                        curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                        axs[i, 2].plot(curve)
                        axs[i, 2].title.set_text('lsc loss')

                    if 'LSC_norms' in k:
                        curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                        axs[i, 3].plot(curve)
                        axs[i, 3].title.set_text('lsc norm')

                    if 'sparse_categorical_accuracy list' == k:
                        axs[i, 4].plot(row[k])
                        axs[i, 4].title.set_text('acc')

                    if 'val_sparse_categorical_accuracy list' == k:
                        axs[i, 4].plot(row[k], '--')
                        axs[i, 4].title.set_text('acc')

                    if 'loss list' == k:
                        axs[i, 5].plot(row[k])
                        axs[i, 5].title.set_text('loss')

                    if 'val_loss list' == k:
                        axs[i, 5].plot(row[k], '--')
                        axs[i, 5].title.set_text('loss')

                except Exception as e:
                    print(e)

    plt.show()

columns_containing = ['_var']

new_column_names = {c_name: shorten_losses(c_name) for c_name in df.columns}
df.rename(columns=new_column_names, inplace=True)
df = df.rename(columns={'test_loss': 'test_loss min', 'test_acc': 'test_acc max'})

plot_only = [
    'pretrain_epochs', 'steps_per_epoch', 'seed', 'lr', 'width', 'layers', 'comments', 'val_acc max',
    'acc max', 'test_loss min', 'test_acc max',
    'loss min', 'val_loss min', 'epoch max', 'time_elapsed', 'hostname', 'path'
]

df = df[plot_only]

print(df.columns)
df = df.sort_values(by=metric)
print(df.to_string())

metrics_oi = [
    # 'loss min',
    'val_loss min', 'test_loss min',
    # 'acc max',
    'val_acc max', 'test_acc max']

group_cols = ['lr', 'comments']
counts = df.groupby(group_cols).size().reset_index(name='counts')

metrics_oi = [shorten_losses(m) for m in metrics_oi]
mdf = df.groupby(
    group_cols, as_index=False
).agg({m: ['mean', 'std'] for m in metrics_oi})

for m in metrics_oi:
    mdf['mean_{}'.format(m)] = mdf[m]['mean']
    mdf['std_{}'.format(m)] = mdf[m]['std']
    mdf = mdf.drop([m], axis=1)
mdf = mdf.droplevel(level=1, axis=1)

mdf['counts'] = counts['counts']
mdf = mdf.sort_values(by='mean_' + metric)

print(mdf.to_string())

# sort mdf by lr
mdf = mdf.sort_values(by='lr')

if lrs_plot:
    fig, axs = plt.subplots(1, 1, figsize=(6, 3))
    comments = mdf['comments'].unique()
    # assign a color to each comment
    colors = {c: np.random.rand(3, ) for c in comments}


    def clean_comments(c):
        if c == 'findLSC':
            return 'sub ($L_2$)'
        if 'radius' in c:
            return r'sub ($\rho$)'
        if c == '':
            return 'Glorot'
        if c == 'heinit':
            return 'He'
        c = c.replace('findLSC_', '')
        c = c.replace('npsd', '')
        c = c.replace('2', '')
        return c


    colors = {'findLSC_radius': [0.43365406, 0.83304796, 0.58958684], '': [0.24995383, 0.49626022, 0.35960801],
              'findLSC': [0.74880857, 0.9167003, 0.50021289], 'findLSC_supnpsd2': [0.69663182, 0.25710645, 0.19346206],
              'findLSC_supsubnpsd': [0.2225346, 0.06820208, 0.9836983], 'heinit': [0.96937357, 0.28256986, 0.26486611]}

    # clean_names = {'findLSC_radius': [0.43365406, 0.83304796, 0.58958684], '': 'Glorot',
    #           'findLSC': [0.74880857, 0.9167003, 0.50021289], 'findLSC_supnpsd2': [0.69663182, 0.25710645, 0.19346206],
    #           'findLSC_supsubnpsd': [0.2225346, 0.06820208, 0.9836983], 'heinit': 'He'}

    for c in comments:
        idf = mdf[mdf['comments'] == c]
        ys = idf['mean_' + metric].values
        yerrs = idf['std_' + metric].values
        xs = idf['lr'].values
        axs.plot(xs, ys, color=colors[c], label=clean_comments(c))
        axs.fill_between(xs, ys - yerrs / 2, ys + yerrs / 2, alpha=0.5, color=colors[c])

    # x axis log scale
    plt.xscale('log')
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.legend()

    # for ax in axs.reshape(-1):
    for pos in ['right', 'left', 'bottom', 'top']:
        axs.spines[pos].set_visible(False)
    axs.locator_params(axis='y', nbins=5)

    plt.show()
    plot_filename = f'experiments/ffn_relu.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')
