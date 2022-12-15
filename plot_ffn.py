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
]

plot_norms_evol = False
plot_norms_evol_1 = True
expsid = 'ffnandcnns'  # effnet als ffnandcnns
h5path = os.path.join(EXPERIMENTS, f'summary_{expsid}.h5')

df = experiments_to_pandas(
    h5path=h5path, zips_folder=GEXPERIMENTS, unzips_folder=EXPERIMENTS, experiments_identifier=expsid,
    exclude_files=['cout.txt'], check_for_new=True
)

# select only rows with width 10
df['time_elapsed'] = pd.to_timedelta(df['time_elapsed'], unit='s')

if plot_norms_evol:
    df = df[df['width'] == 128]
    df = df[df['layers'] == 30]

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

plot_only = ['width', 'layers', 'comments', 'val_sparse_categorical_accuracy max', 'sparse_categorical_accuracy max',
             'loss min', 'val_loss min', 'epoch max', 'time_elapsed', 'hostname']

df = df[plot_only]

new_column_names = {c_name: shorten_losses(c_name) for c_name in df.columns}
df.rename(columns=new_column_names, inplace=True)

print(df.to_string())
