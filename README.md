# Deep Asymmetric Multi-Task Feature Learning (Deep-AMTFL)
+ Hae Beom Lee (KAIST), Eunho Yang (KAIST), and Sung Ju Hwang (KAIST)

This is the Tensor-Flow implementation for the paper Deep Asymmetric Multi-Task Feature Learning (ICML 2018) : https://arxiv.org/abs/1708.00260

## Abstract
<img align="right" width="300" src="https://github.com/HaebeomLee/amtfl/blob/master/concept.png">
We propose Deep Asymmetric Multitask Feature Learning (Deep-AMTFL) which can learn deep representations shared across multiple tasks while effectively preventing negative transfer that may happen in the feature sharing process. Specifically, we introduce an asymmetric autoencoder term that allows reliable predictors for the easy tasks to have high contribution to the feature learning while suppressing the influences of unreliable predictors for more difficult tasks. This allows the learning of less noisy representations, and enables unreliable predictors to exploit knowledge from the reliable predictors via the shared latent features. Such asymmetric knowledge transfer through shared features is also more scalable and efficient than inter-task asymmetric transfer. We validate our Deep-AMTFL model on multiple benchmark datasets for multitask learning and image classification, on which it significantly outperforms existing symmetric and asymmetric multitask learning models, by effectively preventing negative transfer in deep feature learning.

## Reference

If you found the provided code useful, please cite our work.

```
@inproceedings{lee2018amtfl,
    author    = {Hae Beom Lee and Eunho Yang and Sung Ju Hwang},
    title     = {Deep Asymmetric Multi-Task Feature Learning},
    booktitle = {ICML},
    year      = {2018}
}
```

### Run examples
1. Modify the ```--mnist_path``` in ```run.sh```
2. Run 
```
./run.sh
```
