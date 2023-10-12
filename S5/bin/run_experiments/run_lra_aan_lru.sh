python run_train.py --C_init=trunc_standard_normal --batchnorm=True --bidirectional=False \
                    --blocks=16 --bsz=32 --d_model=128 --dataset=aan-classification \
                    --dt_global=True --epochs=20 --jax_seed=5464368 --lr_factor=0.5 --n_layers=6 \
                    --opt_config=standard --p_dropout=0.1 --ssm_lr_base=0.001 --ssm_size_base=256 \
                    --warmup_end=1 --weight_decay=0.05 --activation_fn=full_glu