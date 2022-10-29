import os, json, argparse, copy
import numpy as np
import scipy
from scipy import stats

from GenericTools.keras_tools.esoteric_tasks.time_task_redirection import language_tasks, Task
from GenericTools.stay_organized.utils import str2val, NumpyEncoder

from GenericTools.keras_tools.silence_tensorflow import silence_tf
import matplotlib.pyplot as plt

silence_tf()

from sg_design_lif.neural_models.config import default_config
from alif_sg.neural_models.recLSC import apply_LSC

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPS = os.path.join(CDIR, 'experiments')

# start with LSTM on SHD
parser = argparse.ArgumentParser()
parser.add_argument("--steps_per_epoch", default=1, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--time_steps", default=2, type=int)
parser.add_argument("--task_name", default='heidelberg', type=str)
parser.add_argument("--net_name", default='maLSNN', type=str)
parser.add_argument("--findLSC", default=1, type=int)
args = parser.parse_args()

string_args = json.dumps(vars(args), indent=4, cls=NumpyEncoder)
print(string_args)

clean_task_name = {'sl_mnist': 'sl-MNIST', 'wordptb': 'PTB', 'heidelberg': 'SHD'}
clean_net_name = {'LSTM': 'LSTM', 'maLSNN': 'ALIF'}

epochs = 1
batch_size = args.batch_size
steps_per_epoch = args.steps_per_epoch
time_steps = args.time_steps
stack = None
net_name = args.net_name
task_name = args.task_name
comments = '32_embproj_nogradreset_dropout:.3_timerepeat:2_reoldspike_test'
#
# for net_name in ['LSTM', 'maLSNN']:
#     for task_name in ['heidelberg', 'wordptb', 'sl_mnist']:
print(net_name, task_name)
results_filename = os.path.join(EXPS, f'rec_norms_{net_name}_{task_name}_LSC{args.findLSC}.json')
if not os.path.exists(results_filename):

    # task definition
    stack, batch_size, embedding, n_neurons, lr = default_config(
        stack, batch_size, None, None, None, task_name, net_name
    )
    timerepeat = str2val(comments, 'timerepeat', int, default=1)
    maxlen = str2val(comments, 'maxlen', int, default=100)
    comments = str2val(comments, 'maxlen', int, default=maxlen, replace=maxlen)
    train_task_args = dict(timerepeat=timerepeat, epochs=epochs, batch_size=batch_size,
                           steps_per_epoch=steps_per_epoch,
                           name=task_name, train_val_test='train', maxlen=maxlen, comments=comments)
    gen_train = Task(**train_task_args)

    comments += '_batchsize:' + str(batch_size)
    comments = comments if task_name in language_tasks else comments.replace('embproj', 'simplereadout')

    # task_name, net_name, n_neurons, tau, lr, stack,
    # loss_name, embedding, optimizer_name, tau_adaptation, lr_schedule, weight_decay, clipnorm,
    # initializer, comments, in_len, n_in, out_len, n_out, final_epochs,
    initial_state = None
    model_args = dict(
        task_name=task_name, net_name=net_name, n_neurons=n_neurons,
        lr=lr, stack=stack, loss_name='sparse_categorical_crossentropy',
        embedding=embedding, optimizer_name='SGD', lr_schedule='',
        weight_decay=None, clipnorm=None, initializer='glorot_uniform', comments=comments,
        in_len=gen_train.in_len, n_in=gen_train.in_dim, out_len=gen_train.out_len,
        n_out=gen_train.out_dim, final_epochs=epochs,
    )

    weights = None
    if args.findLSC:

        new_model_args = copy.deepcopy(model_args)
        new_comments = new_model_args['comments'] + '_reoldspike'
        new_batch_size = batch_size
        if 'ptb' in task_name:
            new_batch_size = 8
            new_comments = str2val(new_comments, 'batchsize', replace=new_batch_size)

        new_model_args['comments'] = new_comments

        new_task_args = copy.deepcopy(train_task_args)
        new_task_args['batch_size'] = new_task_args['batch_size'] if not 'ptb' in task_name else new_batch_size

        weights, _ = apply_LSC(
            train_task_args=new_task_args, model_args=new_model_args, norm_pow=2, n_samples=-1,
            batch_size=new_batch_size, depth_norm=False, decoder_norm=False, learn=True,
            steps_per_epoch=2,
            time_steps=time_steps
        )

    _, lsc_results = apply_LSC(
        train_task_args=train_task_args, model_args=model_args, norm_pow=2, n_samples=-1,
        batch_size=batch_size, depth_norm=False, decoder_norm=False, learn=False,
        steps_per_epoch=steps_per_epoch,
        time_steps=time_steps, weights=weights
    )

    # json.dump(lsc_results, open(results_filename, "w"), cls=NumpyEncoder)

    string_result = json.dumps(lsc_results, indent=4, cls=NumpyEncoder)
    print(string_result)
    with open(results_filename, "w") as f:
        f.write(string_result)

else:
    with open(results_filename) as f:
        lsc_results = json.load(f)

print(lsc_results.keys())

# plt.rc('text', usetex=True)
fig, axs = plt.subplots(3, 2, figsize=(6, 3), gridspec_kw={'wspace': .2, 'hspace': .5})

for i in [0, 1]:
    layer = i
    norms = [v for k, v in lsc_results['rec_norms'].items() if f'layer {layer}' in k][0]
    norms = np.array(norms).T

    if args.task_name == 'sl_mnist':
        norms = norms[:, 10:]
    else:
        norms = norms[:, 1:]

    means = np.mean(norms, axis=0)
    normalized_norms = norms - means
    t1x = normalized_norms[:, 0]

    # print(normalized_norms.shape)
    time_steps = normalized_norms.shape[1]
    corr = []
    ps = []
    covs = []

    for t in range(time_steps):
        ns = normalized_norms[:, t]
        # print(ns[:11])
        r, p = scipy.stats.pearsonr(ns, t1x)
        c = np.cov(ns, t1x)[0][1]
        covs.append(c)
        corr.append(r)
        ps.append(p)
    axs[0, i].plot(covs)
    axs[1, i].plot(corr)
    axs[2, i].plot(ps, color='r', linestyle='--')

    axs[2, i].set_xlabel('t')
    axs[0, i].set_title(f'layer {i + 1}')
    axs[0, 0].set_ylabel(r'$Covariance$')
    axs[1, 0].set_ylabel(r'$Correlation$')
    axs[2, 0].set_ylabel(r'$p$-$value$')

for ax in axs.reshape(-1):
    for pos in ['right', 'left', 'bottom', 'top']:
        ax.spines[pos].set_visible(False)

# axs[0, 1].tick_params(labelleft=False, left=False)

fig.align_ylabels(axs[:, 0])
fig.suptitle(f'{clean_net_name[net_name]} on {clean_task_name[task_name]}', y=1.1, fontsize=14)

pathplot = os.path.join(EXPS, f'rec_norms_{net_name}_{task_name}_LSC{args.findLSC}.png')
fig.savefig(pathplot, bbox_inches='tight')
plt.show()
