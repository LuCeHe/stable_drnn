# Stabilizing RNN Gradients through Pre-training

This is the official repository of the [Stabilizing RNN Gradients through Pre-training](https://arxiv.org/abs/2308.12075) 
article, submitted to IEEE.


![Drag Racing](tools/lscs.png)

The FFN experiments are done with the ```training_ffns.py``` script while the RNN experiments
are done with the ```training_rnns.py``` script. Run ```plot_gradient_grid.py``` to generate
Figure 1 a), ```plot_binomial.py``` to generate Figure 1 b) and ```plot_pascalrnn.py``` 
to generate Figure 1 c).

For Figure 2, run as one line


```
python drnn_stability/training_ffns.py
     --depth=30 --width=128 --epochs=50 --steps_per_epoch=-1 --pretrain_epochs=100
     --activation=##act##'
     --dataset=##data##
     --lr=##lr##
     --seed=##seed##
     --comments=##comments##
```


with ```##act##``` one in ```['sin', 'relu', 'cos']```,  ```##data##``` one in ```['mnist', 'cifar10', 'cifar100']```, 
 ```##lr##``` one in ```[1e-2, 3.16e-3, 1e-3, 3.16e-4, 1e-4, 3.16e-5, 1e-5]```, ```##seed##``` one in ```list(range(4))```,
and ```##comments##``` one in ```['findLSC_radius_adabelief_pretrained_onlypretrain']```, and when pretraining has finished successfully,
you run the same hyperparams but this time with 
```##comments##``` as one in ```['findLSC_radius_adabelief_onlyloadpretrained', '_adabelief_onlyloadpretrained', 'heinit_adabelief_onlyloadpretrained']```.

