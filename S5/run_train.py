import argparse, os, time, string, random, shutil, json, socket, re

from alif_sg.S5.s5.dataloaders.base import default_cache_path
from pyaromatics.stay_organized.utils import NumpyEncoder, str2val
from s5.utils.util import str2bool
from s5.train import train
from s5.dataloading import Datasets

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.abspath(os.path.join(CDIR, '..', 'experiments'))
named_tuple = time.localtime()  # get struct_time
time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)

characters = string.ascii_letters + string.digits
random_string = ''.join(random.choice(characters) for i in range(5))
EXPERIMENT = os.path.join(EXPERIMENTS, time_string + random_string + '_s5lru')

os.makedirs(EXPERIMENTS, exist_ok=True)
os.makedirs(EXPERIMENT, exist_ok=True)

if __name__ == "__main__":
    time_start = time.perf_counter()

    parser = argparse.ArgumentParser()

    parser.add_argument("--comments", type=str, default='pretrain', help="String for extra behaviours")
    parser.add_argument("--stop_time", default=600, type=int, help="Stop time")

    parser.add_argument("--USE_WANDB", type=str2bool, default=False,
                        help="log with wandb?")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="wandb entity name, e.g. username")
    parser.add_argument("--dir_name", type=str, default=str(default_cache_path),
                        help="name of directory where data is cached")
    parser.add_argument("--exp_dir", type=str, default=str(EXPERIMENT),
                        help="name of directory where data is cached")
    parser.add_argument("--dataset", type=str, choices=Datasets.keys(),
                        default='cifar-classification',
                        help="dataset name")

    # Model Parameters
    parser.add_argument("--n_layers", type=int, default=1,
                        help="Number of layers in the network")
    parser.add_argument("--d_model", type=int, default=256,
                        help="Number of features, i.e. H, "
                             "dimension of layer inputs/outputs")
    parser.add_argument("--ssm_size_base", type=int, default=192,
                        help="SSM Latent size, i.e. P")
    parser.add_argument("--blocks", type=int, default=12,
                        help="How many blocks, J, to initialize with")
    parser.add_argument("--C_init", type=str, default="lecun_normal",
                        choices=["trunc_standard_normal", "lecun_normal", "complex_normal"],
                        help="Options for initialization of C: \\"
                             "trunc_standard_normal: sample from trunc. std. normal then multiply by V \\ " \
                             "lecun_normal sample from lecun normal, then multiply by V\\ " \
                             "complex_normal: sample directly from complex standard normal")
    parser.add_argument("--discretization", type=str, default="zoh", choices=["zoh", "bilinear"])
    parser.add_argument("--mode", type=str, default="pool", choices=["pool", "last"],
                        help="options: (for classification tasks) \\" \
                             " pool: mean pooling \\" \
                             "last: take last element")
    parser.add_argument("--activation_fn", default="half_glu2", type=str,
                        choices=["full_glu", "half_glu1", "half_glu2", "gelu"])
    parser.add_argument("--conj_sym", type=str2bool, default=True,
                        help="whether to enforce conjugate symmetry")
    parser.add_argument("--clip_eigs", type=str2bool, default=False,
                        help="whether to enforce the left-half plane condition")
    parser.add_argument("--bidirectional", type=str2bool, default=True,
                        help="whether to use bidirectional model")
    parser.add_argument("--dt_min", type=float, default=0.001,
                        help="min value to sample initial timescale params from")
    parser.add_argument("--dt_max", type=float, default=0.1,
                        help="max value to sample initial timescale params from")

    # Optimization Parameters
    parser.add_argument("--prenorm", type=str2bool, default=True,
                        help="True: use prenorm, False: use postnorm")
    parser.add_argument("--batchnorm", type=str2bool, default=True,
                        help="True: use batchnorm, False: use layernorm")
    parser.add_argument("--bn_momentum", type=float, default=0.95,
                        help="batchnorm momentum")
    parser.add_argument("--bsz", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=2, help="max number of epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=2,
                        help="max number steps per epoch")
    parser.add_argument("--early_stop_patience", type=int, default=30,
                        help="number of epochs to continue training when val loss plateaus")
    parser.add_argument("--ssm_lr_base", type=float, default=1e-3,
                        help="initial ssm learning rate")
    parser.add_argument("--lr_factor", type=float, default=4.2,
                        help="global learning rate = lr_factor*ssm_lr_base")
    parser.add_argument("--dt_global", type=str2bool, default=False,
                        help="Treat timescale parameter as global parameter or SSM parameter")
    parser.add_argument("--lr_min", type=float, default=0,
                        help="minimum learning rate")
    parser.add_argument("--cosine_anneal", type=str2bool, default=True,
                        help="whether to use cosine annealing schedule")
    parser.add_argument("--warmup_end", type=int, default=1,
                        help="epoch to end linear warmup")
    parser.add_argument("--lr_patience", type=int, default=1000000,
                        help="patience before decaying learning rate for lr_decay_on_val_plateau")
    parser.add_argument("--reduce_factor", type=float, default=1.0,
                        help="factor to decay learning rate for lr_decay_on_val_plateau")
    parser.add_argument("--p_dropout", type=float, default=0.1,
                        help="probability of dropout")
    parser.add_argument("--weight_decay", type=float, default=0.07,
                        help="weight decay value")
    parser.add_argument("--opt_config", type=str, default="standard", choices=['standard',
                                                                               'BandCdecay',
                                                                               'BfastandCdecay',
                                                                               'noBCdecay'],
                        help="Opt configurations: \\ " \
                             "standard:       no weight decay on B (ssm lr), weight decay on C (global lr) \\" \
                             "BandCdecay:     weight decay on B (ssm lr), weight decay on C (global lr) \\" \
                             "BfastandCdecay: weight decay on B (global lr), weight decay on C (global lr) \\" \
                             "noBCdecay:      no weight decay on B (ssm lr), no weight decay on C (ssm lr) \\")
    parser.add_argument("--jax_seed", type=int, default=1919,
                        help="seed randomness")

    # LRU Parameters
    parser.add_argument("--lru", type=str2bool, default=False,
                        help="True: use LRU, False: don't use LRU")
    parser.add_argument("--r_min", type=float, default=0.5, help="|lambda|_min for LRU")
    parser.add_argument("--r_max", type=float, default=0.99, help="|lambda|_max for LRU")

    args = parser.parse_args()

    args.time_start = time_start

    if 'default' in args.comments:
        filename = f"run_lra_{args.dataset.replace('-classification', '')}.sh"
        if 'defaultlru' in args.comments:
            filename = filename.replace('.sh', '_lru.sh')
        print('Loading default args from ' + filename)
        path_default = os.path.join(CDIR, 'bin', 'run_experiments', filename)

        with open(path_default, 'r') as f:
            outp = f.read().replace('\n', '').replace('\\', '')
        outp = re.sub(' +', ' ', outp)

        dargs = outp.split(' --')[1:]
        # default_bsz = False if not args.bsz < 0 else True
        for arg in dargs:
            if not 'bsz' in arg and not 'jax_seed' in arg and not 'epochs' in arg:
                arg = arg.split('=')
                arg_name = arg[0]
                arg_value = arg[1]
                dtype = type(getattr(args, arg_name))
                dtype = str(dtype).split("'")[1]
                arg_value = eval(f"{dtype}('{arg_value}')")
                setattr(args, arg_name, arg_value)

    string_args = json.dumps(vars(args), indent=4, cls=NumpyEncoder)
    print(string_args)
    results_filename = os.path.join(EXPERIMENT, 'args.json')
    with open(results_filename, "w") as f:
        f.write(string_args)

    lr = str2val(args.comments, 'lr', float, default=args.ssm_lr_base)
    args.ssm_lr_base = lr

    results = train(args)
    time_elapsed = (time.perf_counter() - time_start)
    print('All done, in ' + str(time_elapsed) + 's')

    results.update(time_elapsed=time_elapsed)
    results.update(hostname=socket.gethostname())

    string_result = json.dumps(results, indent=4, cls=NumpyEncoder)
    path = os.path.join(EXPERIMENT, 'results.txt')
    with open(path, "w") as f:
        f.write(string_result)

    shutil.make_archive(EXPERIMENT, 'zip', EXPERIMENT)
