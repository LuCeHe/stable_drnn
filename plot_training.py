import os, json, copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib as mpl

from GenericTools.stay_organized.mpl_tools import load_plot_settings
mpl, pd = load_plot_settings(mpl=mpl, pd=pd)

import matplotlib.pyplot as plt
from datetime import timedelta, datetime

from GenericTools.stay_organized.plot_tricks import large_num_to_reasonable_string

FMT = '%Y-%m-%dT%H:%M:%S'
from GenericTools.stay_organized.unzip import unzip_good_exps

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
GEXPERIMENTS = os.path.join(CDIR, 'good_experiments')

CSVPATH = os.path.join(EXPERIMENTS, 'summary.h5')
HSITORIESPATH = os.path.join(EXPERIMENTS, 'histories.json')

plot_lsc_vs_naive = False
plot_dampenings_and_betas = False

task_name = 'ps_mnist'  # heidelberg wordptb sl_mnist all ps_mnist

# sparse_mode_accuracy sparse_categorical_crossentropy bpc sparse_mode_accuracy_test_10
# val_sparse_mode_accuracy
metric = 'val_sparse_mode_accuracy'
optimizer_name = 'SWAAdaBelief'  # SGD SWAAdaBelief
metrics_oi = ['val_sparse_mode_accuracy', 'val_perplexity', 'val_sparse_categorical_crossentropy',
              'test_sparse_mode_accuracy']

columns_to_remove = [
    'heaviside', '_test', 'epoch', 'weight', 'sLSTM_factor', 'save_model', 'clipnorm', 'GPU', 'batch_size',
    'continue_training', 'embedding', 'lr_schedule', 'loss_name', 'lr', 'net_name', 'seed', 'stack', 'stop_time',
    'convergence', 'n_neurons', 'optimizer_name'
]

def shorten_losses(name):
    shorter_name = name.replace('sparse_', '').replace('categorical_', '').replace('accuracy', 'acc').replace(
            'crossentropy', 'xe').replace('perplexity', 'ppl')
    # shorter_name = name
    return shorter_name

if not os.path.exists(CSVPATH):

    ds = unzip_good_exps(
        GEXPERIMENTS, EXPERIMENTS,
        exp_identifiers=[''], except_folders=[],
        unzip_what=['history.json', 'config.json', 'run.json', 'results.json']
    )

    histories = {}
    df = pd.DataFrame(columns=[])

    for d in tqdm(ds, desc='Creating pandas'):
        history_path = os.path.join(d, 'other_outputs', 'history.json')
        config_path = os.path.join(d, '1', 'config.json')
        run_path = os.path.join(d, '1', 'run.json')
        results_path = os.path.join(d, 'other_outputs', 'results.json')

        with open(config_path) as f:
            config = json.load(f)

        with open(history_path) as f:
            history = json.load(f)

        with open(run_path) as f:
            run = json.load(f)

        with open(results_path) as f:
            some_results = json.load(f)

        results = {}
        results.update(config.items())
        results.update(some_results.items())

        what = lambda k, v: np.nanmax(v) if 'acc' in k else np.nanmin(v)
        results.update({k: what(k, v) for k, v in history.items()})
        results.update({'d': d})

        results.update({'duration_experiment':
                            datetime.strptime(run['stop_time'].split('.')[0], FMT) - datetime.strptime(
                                run['start_time'].split('.')[0], FMT)
                        })

        results['n_params'] = large_num_to_reasonable_string(results['n_params'], 1)

        small_df = pd.DataFrame([results])
        df = df.append(small_df)
        histories[d] = {k: v for k, v in history.items()}

    df = df.sort_values(by='comments')

    df.to_hdf(CSVPATH, key='df', mode='w')
    json.dump(histories, open(HSITORIESPATH, "w"))
else:
    df = pd.read_hdf(CSVPATH, 'df')  # load it
    with open(HSITORIESPATH) as f:
        histories = json.load(f)


# df = df[(df['d'].str.contains('2022-07-28--')) | (df['d'].str.contains('2022-07-29--'))]
df = df[(df['d'].str.contains('2022-08-02--')) ]
df = df.sort_values(by=metric)

for c_name in columns_to_remove:
    df = df[df.columns.drop(list(df.filter(regex=c_name)))]

new_column_names = {c_name: shorten_losses(c_name) for c_name in df.columns}

df.rename(columns=new_column_names, inplace=True)
df = df[[c for c in df if c not in ['d', 'duration_experiment']] + ['d', 'duration_experiment']]

metrics_oi = [shorten_losses(m) for m in metrics_oi]
mdf = df.groupby(
    ['task_name', 'initializer', 'comments'], as_index=False
).agg({m: ['mean', 'std'] for m in metrics_oi})

for m in metrics_oi:
    mdf['mean_{}'.format(m)] = mdf[m]['mean']
    mdf['std_{}'.format(m)] = mdf[m]['std']
    mdf = mdf.drop([m], axis=1)

mdf = mdf.sort_values(by='mean_val_mode_acc')


print(df.to_string())
print(mdf.to_string())

if plot_lsc_vs_naive:

    from matplotlib.lines import Line2D

    if task_name == 'all':
        tasks = np.unique(df['task_name'])
    else:
        tasks = [task_name]

    # idf = df[df['optimizer_name'].str.contains(optimizer_name)]
    idf = copy.deepcopy(df)
    idf['comments'] = idf['comments'].str.replace('_timerepeat:2', '')
    idf['comments'] = idf['comments'].str.replace('timerepeat:2', '')

    for task in tasks:
        iidf = idf[idf['task_name'].str.contains(task)]
        # idf = idf[idf['d'].str.contains('2022-07-06--')]
        # idf = idf[idf['epochs'].eq(1000)]

        print(iidf.to_string())
        n_plots = 10
        colors_for_type = {
            'LSC1': 'Blues',
            # 'dampening:1.': 'Oranges',
            'randominit': 'Reds',
            'lsc1': 'Greens',
            # 'LSC_dampening:1.': 'Purples',
            # 'original': 'Purples',
            '': 'Purples',
            'LSC2': 'Oranges',
            # 'LSC2_ingain:1.414': 'Greens',
        }

        # types = ['LSC', 'dampening:1.', 'randominit', 'lscc', 'LSC_dampening:1.', 'original', '']
        types = colors_for_type.keys()
        m = metric.replace('val_', '')
        # types = ['LSC', 'randominit', 'original']
        fig, axs = plt.subplots(1, 2, figsize=(6, 2), sharey=True, gridspec_kw={'wspace': .05})
        for i in range(2):
            for comment in types :
                iiidf = iidf[iidf['comments'].eq(comment.replace('_timerepeat:2', ''))]
                print(iiidf.to_string())
                # print(colors[comment])
                cmap = plt.cm.get_cmap(colors_for_type[comment])
                colors = cmap(np.arange(iiidf.shape[0]) / iiidf.shape[0])
                for j, (_, row) in enumerate(iiidf.iterrows()):
                    d = row['d']
                    h = histories[d][m if i == 0 else 'val_' + m]
                    axs[i].plot(h, color=colors[j], label=comment)

        axs[0].set_title('train ' + task)
        axs[1].set_title('validation')
        axs[0].set_ylabel('accuracy')
        axs[1].set_xlabel('training epoch')

        custom_lines = [Line2D([0], [0], color=plt.cm.get_cmap(colors_for_type[type])(0.5), lw=4) for type in types]

        axs[1].legend(custom_lines, [t.replace('init', '').replace('original', 'reference') for t in types],
                      loc='lower right', framealpha=0.9)

        for ax in axs.reshape(-1):
            for pos in ['right', 'left', 'bottom', 'top']:
                ax.spines[pos].set_visible(False)

        # axs[1].axes.yaxis.set_ticklabels([])
        axs[1].tick_params(labelleft=False, left=False)
        pathplot = os.path.join(CDIR, 'experiments', f'lscvsrandom_{task}.png')
        fig.savefig(pathplot, bbox_inches='tight')

        plt.show()

if plot_dampenings_and_betas:
    init = 'LSC2'  # lsc LSC LSC2
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    idf = df[df['task_name'].str.contains(task_name)]
    idf = idf[idf['optimizer_name'].str.contains(optimizer_name)]
    iidf = idf[idf['comments'].str.contains(init + '_beta')]  # cdr gra blg
    iidf['betas'] = iidf['comments'].str.replace(init + '_beta:', '').values.astype(float)
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

    metric = metric.replace('val_', '')
    means = mdf['mean_val_' + metric]
    stds = mdf['std_val_' + metric]
    betas = mdf['betas']

    axs[0].plot(betas, means, color=color)
    axs[0].fill_between(betas, means - stds, means + stds, alpha=0.5, color=color)

    idf = df[df['task_name'].str.contains(task_name)]
    idf = idf[idf['optimizer_name'].str.contains(optimizer_name)]
    iidf = idf[idf['comments'].str.contains(init + '_dampening')]  # cdr gra blg
    iidf['dampening'] = iidf['comments'].str.replace(init + '_dampening:', '').values.astype(float)
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
