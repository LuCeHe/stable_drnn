import os, json, argparse

from GenericTools.keras_tools.esoteric_tasks.time_task_redirection import language_tasks, Task
from GenericTools.stay_organized.utils import str2val, NumpyEncoder

from sg_design_lif.neural_models.config import default_config
from alif_sg.neural_models.recLSC import apply_LSC

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPS = os.path.join(CDIR, 'experiments')

# start with LSTM on SHD

parser = argparse.ArgumentParser()
parser.add_argument("--steps_per_epoch", default=1, type=int, help="Batch size")
parser.add_argument("--time_steps", default=1, type=int, help="timesteps")
parser.add_argument("--task_name", default='heidelberg', type=str)
parser.add_argument("--net_name", default='maLSNN', type=str)
args = parser.parse_args()


epochs = 1
batch_size = 1
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
results_filename = os.path.join(EXPS, f'rec_norms_{net_name}_{task_name}.json')
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

    _, lsc_results = apply_LSC(
        train_task_args=train_task_args, model_args=model_args, norm_pow=2, n_samples=-1,
        batch_size=batch_size, depth_norm=False, decoder_norm=False, learn=False,
        steps_per_epoch=steps_per_epoch,
        time_steps=time_steps
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

layer = 1
norms = [v for k, v in lsc_results['rec_norms'].items() if f'layer {layer}' in k]

print([len(n) for n in norms])
import numpy as np

norms = np.array(norms)
means = np.mean(norms, axis=0)
normalized_norms = norms - means
t1x = normalized_norms[:, 0]
cov_i = normalized_norms * t1x[..., None]
cov = np.mean(cov_i, axis=0)

print(norms.shape, norms[0].shape, t1x.shape, cov_i.shape)
print(cov)

# c = (norms-)
