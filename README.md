# Environment requirements
```
python>=3.8
pytorch==1.12
```

# Command
```
python exp/cifar10_emcmc.py
python exp/cifar100_emcmc.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python exp/imagenet_emcmc.py
```

# Citation
```
@inproceedings{lientropy,
  title={Entropy-MCMC: Sampling from Flat Basins with Ease},
  author={Li, Bolian and Zhang, Ruqi},
  booktitle={The Twelfth International Conference on Learning Representations}
}
```
