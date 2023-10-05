python run_train.py --activation_fn=full_glu \
                    --batchnorm=True --bsz=50 \
                    --d_model=256 --dataset=imdb-classification \
                    --dt_global=True --epochs=35 --jax_seed=8825365 --lr_factor=0.1 \
                    --n_layers=6 --p_dropout=0.1 --ssm_lr_base=0.004 \
                    --ssm_size_base=192 --warmup_end=0 --weight_decay=0.05