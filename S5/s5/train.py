import os, json, time
from functools import partial
from jax import random
import jax.numpy as np
from jax.scipy.linalg import block_diag

from pyaromatics.stay_organized.utils import NumpyEncoder
from .layers import SequenceLayer
from .train_helpers import create_train_state, reduce_lr_on_plateau, \
    linear_warmup, cosine_annealing, constant_lr, train_epoch, validate
from .dataloading import Datasets
from .seq_model import BatchClassificationModel, RetrievalModel
from .ssm import init_S5SSM
from .ssm_init import make_DPLR_HiPPO
# from alif_sg.minimal_LRU_modified.lru.model import LRU
from alif_sg.S5.jax_pretrain import pretrain
from alif_sg.S5.s5.lru_model import LRU


def train(args):
    """
    Main function to train over a certain number of epochs
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    ssm_lr = args.ssm_lr_base
    lr = args.lr_factor * ssm_lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    create_dataset_fn = Datasets[args.dataset]

    # Dataset dependent logic
    if args.dataset in ["imdb-classification", "listops-classification", "aan-classification"]:
        padded = True
        if args.dataset in ["aan-classification"]:
            # Use retreival model for document matching
            retrieval = True
            print("Using retrieval model for document matching")
        else:
            retrieval = False

    else:
        padded = False
        retrieval = False

    # For speech dataset
    if args.dataset in ["speech35-classification"]:
        speech = True
        print("Will evaluate on both resolutions for speech task")
    else:
        speech = False

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = \
        create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    d_model = args.d_model
    if args.lru and not 'variant' in args.comments:
        model_name = "LRU"
        d_hidden = int(args.ssm_size_base * .7)
        if 'lruv2' in args.comments:
            d_model = args.ssm_size_base
            d_hidden = int(args.d_model * .89)
        elif 'lruv3' in args.comments:
            d_hidden = args.ssm_size_base

        # d_hidden = N
        # d_model = H

        lru = partial(
            LRU, d_hidden=d_hidden, d_model=d_model, r_min=args.r_min, r_max=args.r_max, max_phase=args.max_phase
        )
        ssm_init_fn = lru

    elif args.lru and 'variant' in args.comments:

        model_name = "LRUv"
        d_hidden = int(args.ssm_size_base * .7)
        if 'lruv2' in args.comments:
            d_model = args.ssm_size_base
            d_hidden = int(args.d_model * .89)
        elif 'lruv3' in args.comments:
            d_hidden = args.ssm_size_base

        if 'variant1' in args.comments:
            from alif_sg.S5.s5.lru_variants import LRU_real
            rnn = LRU_real
            model_name += '1'
        elif 'variant2' in args.comments:
            from alif_sg.S5.s5.lru_variants import twolru
            rnn = twolru
            model_name += '2'
        else:
            raise NotImplementedError
        # d_hidden = N
        # d_model = H

        lru = partial(
            rnn, d_hidden=d_hidden, d_model=d_model, comments=args.comments
        )
        ssm_init_fn = lru

    else:
        model_name = "S5"

        # determine the size of initial blocks
        ssm_size = args.ssm_size_base
        block_size = int(ssm_size / args.blocks)

        # Initialize state matrix A using approximation to HiPPO-LegS matrix
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

        if args.conj_sym:
            block_size = block_size // 2
            ssm_size = ssm_size // 2

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
        Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
        V = block_diag(*([V] * args.blocks))
        Vinv = block_diag(*([Vc] * args.blocks))

        ssm_init_fn = init_S5SSM(H=args.d_model,
                                 P=ssm_size,
                                 Lambda_re_init=Lambda.real,
                                 Lambda_im_init=Lambda.imag,
                                 V=V,
                                 Vinv=Vinv,
                                 C_init=args.C_init,
                                 discretization=args.discretization,
                                 dt_min=args.dt_min,
                                 dt_max=args.dt_max,
                                 conj_sym=args.conj_sym,
                                 clip_eigs=args.clip_eigs,
                                 bidirectional=args.bidirectional)
    print(f"[*] Starting {model_name} Training on `{args.dataset}` =>> Initializing...")

    if retrieval:
        # Use retrieval head for AAN task
        print("Using Retrieval head for {} task".format(args.dataset))
        model_cls = partial(
            RetrievalModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
            comments=args.comments
        )

    else:
        model_cls = partial(
            BatchClassificationModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
            comments=args.comments
        )

    # initialize training state
    state, n_params = create_train_state(
        model_cls,
        init_rng,
        padded,
        retrieval,
        in_dim=in_dim,
        bsz=args.bsz,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        batchnorm=args.batchnorm,
        opt_config=args.opt_config,
        ssm_lr=ssm_lr,
        lr=lr,
        dt_global=args.dt_global,
        args=args
    )

    results = {}
    if 'pretrain' in args.comments:
        print("[*] Pretraining")
        from flax import linen as nn

        params = state.params
        time_steps = 3
        for li in range(args.n_layers):
            VmappedSL = nn.vmap(
                SequenceLayer,
                in_axes=0, out_axes=0,
                variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
                split_rngs={"params": False, "dropout": True}, axis_name='batch')

            model = partial(
                VmappedSL,
                ssm=ssm_init_fn,
                dropout=args.p_dropout,
                d_model=d_model,
                activation=args.activation_fn,
                prenorm=args.prenorm,
                batchnorm=args.batchnorm,
                bn_momentum=args.bn_momentum,
            )(training=True)

            loss_threshold = 0.01
            new_params, presults = pretrain(
                model, args.jax_seed + li, batch_size=args.ptbsz, pretrain_steps=args.ptsteps,
                time_steps=time_steps, features=d_model, comments=args.comments, ptcomments=args.ptcomments,
                loss_threshold=loss_threshold, ptlr=args.ptlr,
                optimizer=args.ptopt
            )
            presults = {f'l{li}_' + k: v for k, v in presults.items()}

            results.update(presults)
            params['encoder'][f'layers_{li}'] = new_params
        state = state.replace(params=params)

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = 0  # for per step learning rate decay
    steps_per_epoch = int(train_size / args.bsz)

    results.update({
        "n_params": n_params,
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": [],
    })

    train_loss, val_loss, val_acc, test_loss, test_acc = 0, 0, 0, 0, 0
    for epoch in range(args.epochs):
        time_elapsed = (time.perf_counter() - args.time_start)
        if time_elapsed > args.stop_time:
            print("Time limit reached, ending training")
            break
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        if epoch < args.warmup_end:
            print("using linear warmup for epoch {}".format(epoch + 1))
            decay_function = linear_warmup
            end_step = steps_per_epoch * args.warmup_end

        elif args.cosine_anneal:
            print("using cosine annealing for epoch {}".format(epoch + 1))
            decay_function = cosine_annealing
            # for per step learning rate decay
            end_step = steps_per_epoch * args.epochs - (steps_per_epoch * args.warmup_end)
        else:
            print("using constant lr for epoch {}".format(epoch + 1))
            decay_function = constant_lr
            end_step = None

        # TODO: Switch to letting Optax handle this.
        #  Passing this around to manually handle per step learning rate decay.
        lr_params = (decay_function, ssm_lr, lr, step, end_step, args.opt_config, args.lr_min)

        train_rng, skey = random.split(train_rng)
        state, train_loss, step = train_epoch(
            state, skey, model_cls, trainloader, seq_len, in_dim, args.batchnorm, lr_params, args=args
        )

        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_loss, val_acc = validate(
                state, model_cls, valloader, seq_len, in_dim, args.batchnorm, args=args
            )

            print(f"[*] Running Epoch {epoch + 1} Test...")
            test_loss, test_acc = validate(
                state, model_cls, testloader, seq_len, in_dim, args.batchnorm, args=args
            )

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f} -- Test Loss: {test_loss:.5f} --"
                f" Val Accuracy: {val_acc:.4f}"
                f" Test Accuracy: {test_acc:.4f}"
            )

        else:
            # else use test set as validation set (e.g. IMDB)
            print(f"[*] Running Epoch {epoch + 1} Test...")
            val_loss, val_acc = validate(
                state, model_cls, testloader, seq_len, in_dim, args.batchnorm, args=args
            )

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}  -- Test Loss: {val_loss:.5f} --"
                f" Test Accuracy: {val_acc:.4f}"
            )

        # For early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1

        if val_acc > best_acc:
            # Increment counters etc.
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

            # Do some validation on improvement.
            if speech:
                # Evaluate on resolution 2 val and test sets
                print(f"[*] Running Epoch {epoch + 1} Res 2 Validation...")
                val2_loss, val2_acc = validate(state,
                                               model_cls,
                                               aux_dataloaders['valloader2'],
                                               int(seq_len // 2),
                                               in_dim,
                                               args.batchnorm,
                                               step_rescale=2.0)

                print(f"[*] Running Epoch {epoch + 1} Res 2 Test...")
                test2_loss, test2_acc = validate(state, model_cls, aux_dataloaders['testloader2'], int(seq_len // 2),
                                                 in_dim, args.batchnorm, step_rescale=2.0)
                print(f"\n=>> Epoch {epoch + 1} Res 2 Metrics ===")
                print(
                    f"\tVal2 Loss: {val2_loss:.5f} --Test2 Loss: {test2_loss:.5f} --"
                    f" Val Accuracy: {val2_acc:.4f}"
                    f" Test Accuracy: {test2_acc:.4f}"
                )

        # For learning rate decay purposes:
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(input, factor=args.reduce_factor,
                                                             patience=args.lr_patience, lr_min=args.lr_min)

        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        results["train_loss"].append(float(train_loss))
        results["val_loss"].append(float(val_loss))
        results["val_acc"].append(float(val_acc))
        results["test_loss"].append(float(test_loss))
        results["test_acc"].append(float(test_acc))

        string_result = json.dumps(results, indent=4, cls=NumpyEncoder)
        path = os.path.join(args.exp_dir, 'results.txt')
        with open(path, "w") as f:
            f.write(string_result)

        if count > args.early_stop_patience:
            break

    return results
