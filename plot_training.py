import os, json
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GenericTools.stay_organized.unzip import unzip_good_exps

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
GEXPERIMENTS = os.path.join(CDIR, 'good_experiments')

CSVPATH = os.path.join(EXPERIMENTS, 'summary.h5')
HSITORIESPATH = os.path.join(EXPERIMENTS, 'histories.json')

plot_lsc_vs_naive = False
plot_betas = False
plot_dampenings = True

if not os.path.exists(CSVPATH):

    ds = unzip_good_exps(
        GEXPERIMENTS, EXPERIMENTS,
        exp_identifiers=[''], except_folders=[],
        unzip_what=['history.json', 'config']
    )

    histories = {}
    df = pd.DataFrame(columns=[])

    for d in tqdm(ds, desc='Creating pandas'):
        history_path = os.path.join(d, 'other_outputs', 'history.json')
        config_path = os.path.join(d, '1', 'config.json')
        # text_file = os.path.join(d, [l for l in os.listdir(d) if '.txt' in l][0])

        with open(config_path) as f:
            config = json.load(f)

        with open(history_path) as f:
            history = json.load(f)

        results = {}
        results.update({k: v for k, v in config.items()})
        what = lambda k, v: max(v) if 'acc' in k else min(v)
        results.update({k: what(k, np.array(v)[~np.isnan(np.array(v))]) for k, v in history.items()})
        results.update({'d': d})

        small_df = pd.DataFrame([results])

        df = df.append(small_df)
        history = {k.replace('val_', ''): v for k, v in history.items() if 'val' in k}
        histories[d] = history

    df = df.sort_values(by='comments')

    df.to_hdf(CSVPATH, key='df', mode='w')
    json.dump(histories, open(HSITORIESPATH, "w"))
    # print(df.to_string())
else:
    # mdf = pd.read_csv(CSVPATH)
    df = pd.read_hdf(CSVPATH, 'df')  # load it
    with open(HSITORIESPATH) as f:
        histories = json.load(f)

print(df.to_string())

if plot_lsc_vs_naive:
    task_name = 'wordptb'  # heidelberg wordptb
    metric = 'bpc'  # sparse_mode_accuracy sparse_categorical_crossentropy bpc
    idf = df[df['optimizer_name'].str.contains('SWAAdaBelief')]
    idf = idf[idf['task_name'].str.contains(task_name)]

    colors = {
        'LSC': [plt.cm.Greens(x / 6) for x in range(1, 5)],
        'dampening:1.': [plt.cm.Oranges(x / 6) for x in range(1, 5)],
    }

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    for comment in ['LSC', 'dampening:1.']:
        iidf = idf[idf['comments'].eq(comment)]
        # print(iidf.to_string())

        for (_, row), c in zip(iidf.iterrows(), colors[comment]):
            d = row['d']
            h = histories[d][metric]
            axs.plot(h, color=c)

    axs.set_title(task_name)
    plt.show()

if plot_betas:
    task_name = 'wordptb'  # heidelberg wordptb
    metric = 'bpc'  # sparse_mode_accuracy sparse_categorical_crossentropy bpc
    # idf = df[df['optimizer_name'].str.contains('SWAAdaBelief')]
    idf = df[df['task_name'].str.contains(task_name)]
    iidf = idf[idf['comments'].str.contains('LSC_beta')]  # cdr gra blg

    metrics_oi = ['val_sparse_mode_accuracy', 'bpc', 'val_sparse_categorical_crossentropy']

    mdf = iidf.groupby(
        ['net_name', 'task_name', 'initializer', 'comments'], as_index=False
    ).agg({m: ['mean', 'std'] for m in metrics_oi})

    for metric in metrics_oi:
        mdf['mean_{}'.format(metric)] = mdf[metric]['mean']
        mdf['std_{}'.format(metric)] = mdf[metric]['std']
        mdf = mdf.drop([metric], axis=1)

    mdf['betas'] = mdf['comments'].str.replace('LSC_beta:', '').values.astype(float)
    print(mdf.to_string())

    # colors = [plt.cm.Oranges(x / 6) for x in range(1, 5)]
    color = plt.cm.Oranges(3 / 6)
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    means = mdf['mean_' + metric]
    stds= mdf['std_' + metric]
    betas = mdf['betas']
    axs.plot(betas, means, color=color)
    axs.fill_between(betas, means - stds, means + stds, alpha=0.5, color=color)

    axs.set_title(task_name)
    plt.show()


if plot_dampenings:
    task_name = 'wordptb'  # heidelberg wordptb
    metric = 'bpc'  # sparse_mode_accuracy sparse_categorical_crossentropy bpc
    # idf = df[df['optimizer_name'].str.contains('SWAAdaBelief')]
    idf = df[df['task_name'].str.contains(task_name)]
    iidf = idf[idf['comments'].str.contains('LSC_dampening')]  # cdr gra blg

    metrics_oi = ['val_sparse_mode_accuracy', 'bpc', 'val_sparse_categorical_crossentropy']

    mdf = iidf.groupby(
        ['net_name', 'task_name', 'initializer', 'comments'], as_index=False
    ).agg({m: ['mean', 'std'] for m in metrics_oi})

    for metric in metrics_oi:
        mdf['mean_{}'.format(metric)] = mdf[metric]['mean']
        mdf['std_{}'.format(metric)] = mdf[metric]['std']
        mdf = mdf.drop([metric], axis=1)

    mdf['dampenings'] = mdf['comments'].str.replace('LSC_dampening:', '').values.astype(float)
    print(mdf.to_string())

    # colors = [plt.cm.Oranges(x / 6) for x in range(1, 5)]
    color = plt.cm.Blues(3 / 6)
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    means = mdf['mean_' + metric]
    stds= mdf['std_' + metric]
    dampenings = mdf['dampenings']
    axs.plot(dampenings, means, color=color)
    axs.fill_between(dampenings, means - stds, means + stds, alpha=0.5, color=color)

    axs.set_title(task_name)
    plt.show()
