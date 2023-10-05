python run_train.py --C_init=lecun_normal --batchnorm=True --bidirectional=False \
                    --bsz=50 --d_model=512 --ssm_size_base=384 --dataset=lra-cifar-classification \
                    --epochs=250 --jax_seed=16416 --lr_factor=0.25 --n_layers=6 --opt_config=BfastandCdecay \
                    --p_dropout=0.1 --warmup_end=1 --weight_decay=0.05 \
                    --ssm_lr_base=0.004 --activation_fn=full_glu