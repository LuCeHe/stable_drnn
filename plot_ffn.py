import os
import numpy as np
import matplotlib.pyplot as plt

from stay_organized.pandardize import experiments_to_pandas

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
GEXPERIMENTS = [
    os.path.join(CDIR, 'good_experiments'),
]


expsid = 'ffnandcnns'  # effnet als ffnandcnns
h5path = os.path.join(EXPERIMENTS, f'summary_{expsid}.h5')

df = experiments_to_pandas(
    h5path=h5path, zips_folder=GEXPERIMENTS, unzips_folder=EXPERIMENTS, experiments_identifier=expsid,
    exclude_files=['cout.txt'], check_for_new=False
)
print(df.shape)
print(df['comments'])
print(list(df.columns))

for _, row in df.iterrows():

    fig, axs = plt.subplots(2, 3)
    fig.suptitle(row['comments'])

    for k in row.keys():
        try:
            if '_mean' in k:
                print(row['comments'],row['path'], k)
                print(row[k])
                curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                axs[0, 0].plot(curve)
                axs[0, 0].title.set_text('mean')

            if '_var' in k:
                print(row[k])
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

        except Exception as e:
            print(e)

    plt.show()