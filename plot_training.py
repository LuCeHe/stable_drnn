import os, json
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

FMT = '%Y-%m-%dT%H:%M:%S'
from GenericTools.stay_organized.unzip import unzip_good_exps

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
GEXPERIMENTS = os.path.join(CDIR, 'good_experiments')

CSVPATH = os.path.join(EXPERIMENTS, 'summary.h5')
HSITORIESPATH = os.path.join(EXPERIMENTS, 'histories.json')

plot_lsc_vs_naive = True
plot_betas = False
plot_dampenings = False
plot_dampenings_and_betas = False

task_name = 'heidelberg'  # heidelberg wordptb sl_mnist
metric = 'sparse_mode_accuracy'  # sparse_mode_accuracy sparse_categorical_crossentropy bpc
optimizer_name = 'SWAAdaBelief'  # SGD SWAAdaBelief
metrics_oi = ['val_sparse_mode_accuracy', 'bpc', 'val_sparse_categorical_crossentropy']

if not os.path.exists(CSVPATH):

    ds = unzip_good_exps(
        GEXPERIMENTS, EXPERIMENTS,
        exp_identifiers=[''], except_folders=[],
        unzip_what=['history.json', 'config', 'run.json']
    )

    histories = {}
    df = pd.DataFrame(columns=[])

    for d in tqdm(ds, desc='Creating pandas'):
        history_path = os.path.join(d, 'other_outputs', 'history.json')
        config_path = os.path.join(d, '1', 'config.json')
        run_path = os.path.join(d, '1', 'run.json')
        # text_file = os.path.join(d, [l for l in os.listdir(d) if '.txt' in l][0])

        with open(config_path) as f:
            config = json.load(f)

        with open(history_path) as f:
            history = json.load(f)

        with open(run_path) as f:
            run = json.load(f)

        results = {}
        results.update({k: v for k, v in config.items()})
        what = lambda k, v: max(v) if 'acc' in k else min(v)
        results.update({k: what(k, np.array(v)[~np.isnan(np.array(v))]) for k, v in history.items()})
        results.update({'d': d})

        results.update({'duration_experiment':
                            datetime.strptime(run['stop_time'].split('.')[0], FMT) - datetime.strptime(
                                run['start_time'].split('.')[0], FMT)
                        })

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
    idf = df[df['optimizer_name'].str.contains(optimizer_name)]
    idf = idf[idf['task_name'].str.contains(task_name)]
    idf = idf[idf['d'].str.contains('2022-07-06--')]
    idf = idf[idf['epochs'].eq(1000)]

    print(idf.to_string())

    colors = {
        'LSC': [plt.cm.Greens(x / 6) for x in range(1, 5)],
        'dampening:1.': [plt.cm.Oranges(x / 6) for x in range(1, 5)],
        'randominit':[plt.cm.Reds(x / 6) for x in range(1, 5)],
    }

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    for comment in ['LSC', 'dampening:1.', 'randominit']:
        iidf = idf[idf['comments'].eq(comment)]
        # print(iidf.to_string())

        for (_, row), c in zip(iidf.iterrows(), colors[comment]):
            d = row['d']
            h = histories[d][metric]
            axs.plot(h, color=c, label=comment)

    axs.set_title(task_name)
    axs.set_ylabel(metric)
    plt.legend()

    pathplot = os.path.join(CDIR, 'experiments', 'lscvsrandom.png')
    fig.savefig(pathplot, bbox_inches='tight')

    plt.show()

if plot_betas:
    idf = df[df['task_name'].str.contains(task_name)]
    idf = idf[idf['optimizer_name'].str.contains(optimizer_name)]
    iidf = idf[idf['comments'].str.contains('LSC_beta')]  # cdr gra blg
    iidf['betas'] = iidf['comments'].str.replace('LSC_beta:', '').values.astype(float)
    iidf = iidf.sort_values(by='betas')

    colors = [plt.cm.Greens(x / (len(iidf) + 1)) for x in range(1, len(iidf) + 1)]

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    for (_, row), c in zip(iidf.iterrows(), colors):
        d = row['d']
        h = histories[d][metric]
        axs.plot(h, color=c, label=row['betas'])

    axs.set_title(task_name)
    axs.set_ylabel(metric)
    plt.legend()
    plt.show()

    mdf = iidf.groupby(
        ['net_name', 'task_name', 'initializer', 'betas'], as_index=False
    ).agg({m: ['mean', 'std'] for m in metrics_oi})

    for m in metrics_oi:
        mdf['mean_{}'.format(m)] = mdf[m]['mean']
        mdf['std_{}'.format(m)] = mdf[m]['std']
        mdf = mdf.drop([m], axis=1)
    mdf = mdf.sort_values(by='betas')

    print(mdf.to_string())

    color = plt.cm.Oranges(3 / 6)
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    means = mdf['mean_val_' + metric]
    stds = mdf['std_val_' + metric]
    betas = mdf['betas']
    axs.plot(betas, means, color=color)
    axs.fill_between(betas, means - stds, means + stds, alpha=0.5, color=color)

    axs.set_ylabel(metric)
    axs.set_title(task_name)
    plt.show()

if plot_dampenings:
    idf = df[df['task_name'].str.contains(task_name)]
    idf = idf[idf['optimizer_name'].str.contains(optimizer_name)]
    # idf = idf[idf['d'].str.contains('2022-07-04--')]
    iidf = idf[idf['comments'].str.contains('LSC_dampening')]  # cdr gra blg
    iidf['dampening'] = iidf['comments'].str.replace('LSC_dampening:', '').values.astype(float)
    iidf = iidf.sort_values(by='dampening')

    colors = [plt.cm.Greens(x / (len(iidf) + 1)) for x in range(1, len(iidf) + 1)]

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    for (_, row), c in zip(iidf.iterrows(), colors):
        d = row['d']
        h = histories[d][metric]
        axs.plot(h, color=c, label=row['dampening'])

    axs.set_title(task_name)
    axs.set_ylabel(metric)
    plt.legend()
    plt.show()

    mdf = iidf.groupby(
        ['net_name', 'task_name', 'initializer', 'dampening'], as_index=False
    ).agg({m: ['mean', 'std'] for m in metrics_oi})

    for m in metrics_oi:
        mdf['mean_{}'.format(m)] = mdf[m]['mean']
        mdf['std_{}'.format(m)] = mdf[m]['std']
        mdf = mdf.drop([m], axis=1)
    mdf = mdf.sort_values(by='dampening')

    print(mdf.to_string())

    color = plt.cm.Oranges(3 / 6)
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    means = mdf['mean_val_' + metric]
    stds = mdf['std_val_' + metric]
    betas = mdf['dampening']
    axs.plot(betas, means, color=color)
    axs.fill_between(betas, means - stds, means + stds, alpha=0.5, color=color)

    axs.set_ylabel(metric)
    axs.set_title(task_name)
    plt.show()

if plot_dampenings_and_betas:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    idf = df[df['task_name'].str.contains(task_name)]
    idf = idf[idf['optimizer_name'].str.contains(optimizer_name)]
    iidf = idf[idf['comments'].str.contains('LSC_beta')]  # cdr gra blg
    iidf['betas'] = iidf['comments'].str.replace('LSC_beta:', '').values.astype(float)
    iidf = iidf.sort_values(by='betas')

    mdf = iidf.groupby(
        ['net_name', 'task_name', 'initializer', 'betas'], as_index=False
    ).agg({m: ['mean', 'std'] for m in metrics_oi})

    for m in metrics_oi:
        mdf['mean_{}'.format(m)] = mdf[m]['mean']
        mdf['std_{}'.format(m)] = mdf[m]['std']
        mdf = mdf.drop([m], axis=1)
    mdf = mdf.sort_values(by='betas')

    print(mdf.to_string())

    color = plt.cm.Oranges(3 / 6)

    means = mdf['mean_val_' + metric]
    stds = mdf['std_val_' + metric]
    betas = mdf['betas']

    axs[0].plot(betas, means, color=color)
    axs[0].fill_between(betas, means - stds, means + stds, alpha=0.5, color=color)

    idf = df[df['task_name'].str.contains(task_name)]
    idf = idf[idf['optimizer_name'].str.contains(optimizer_name)]
    iidf = idf[idf['comments'].str.contains('LSC_dampening')]  # cdr gra blg
    iidf['dampening'] = iidf['comments'].str.replace('LSC_dampening:', '').values.astype(float)
    iidf = iidf.sort_values(by='dampening')

    mdf = iidf.groupby(
        ['net_name', 'task_name', 'initializer', 'dampening'], as_index=False
    ).agg({m: ['mean', 'std'] for m in metrics_oi})

    for m in metrics_oi:
        mdf['mean_{}'.format(m)] = mdf[m]['mean']
        mdf['std_{}'.format(m)] = mdf[m]['std']
        mdf = mdf.drop([m], axis=1)
    mdf = mdf.sort_values(by='dampening')

    color = plt.cm.Oranges(3 / 6)

    means = mdf['mean_val_' + metric]
    stds = mdf['std_val_' + metric]
    dampening = mdf['dampening']
    axs[1].plot(dampening, means, color=color)
    axs[1].fill_between(dampening, means - stds, means + stds, alpha=0.5, color=color)

    axs[0].set_ylabel('accuracy')
    axs[0].set_xlabel(r'$\beta$')
    axs[1].set_xlabel('$\gamma$')

    # LSC values
    # axs.axvline(x=value, color='k', linestyle='--')

    # axs[1].set_ylabel(metric)
    # axs[1].set_title(task_name)

    for ax in axs.reshape(-1):
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

    pathplot = os.path.join(CDIR, 'experiments', 'beta_dampening.png')
    fig.savefig(pathplot, bbox_inches='tight')

    plt.show()
