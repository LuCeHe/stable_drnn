python run_train.py --C_init=lecun_normal --activation_fn=full_glu  --bidirectional=False --batchnorm=True \
                    --bsz=50 --d_model=128 --dataset=listops-classification \
                    --epochs=40 --jax_seed=6554595 --lr_factor=0.5 --n_layers=6 \
                    --p_dropout=0 --ssm_lr_base=0.004 --ssm_size_base=192 --warmup_end=1 --weight_decay=0.05