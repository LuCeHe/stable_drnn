from GenericTools.keras_tools.expose_latent import expose_latent_model

model_args = dict(
    task_name=args.task_name, net_name='maLSNN', n_neurons=n_neurons, tau=tau, lr=0., stack=stack,
    loss_name='sparse_categorical_crossentropy', embedding=embedding, optimizer_name='SWAAdaBelief',
    lr_schedule='', weight_decay=.01 if not 'mnist' in args.task_name else 0., clipnorm=1.,
    initializer='glorot_uniform', comments=base_comments + c,
    in_len=gen_val.in_len, n_in=gen_val.in_dim, out_len=gen_val.out_len,
    n_out=gen_val.out_dim, tau_adaptation=tau_adaptation, final_epochs=gen_val.epochs,
)
model = build_model(**model_args)
new_model = expose_latent_model(model, include_layers=[layer_identifier], idx=output_index)
