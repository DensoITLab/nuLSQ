# nuLSQ
This repo contains the code of the following paper accepeted by ACCV 2024


## How to Run Scripts with Your settings
In this program, you train the model under your own configurations. You can modify most of the configuration settings inside `./config/*yaml`  file, and specify the arguments given as follows.

| Arguments | Description |
|---------------------|---------------------|
| --train_id | training id for results management |
|--num_bits | Bit-width |
| --x_quantizer, --w_quantizer| quantizers used in activation, weights ( MinMax_quantizer, LSQ_quantizer, LCQ_quantizer, APoT_quantizer, Positive_nuLSQ_quantizer,Symmetric_nuLSQ_quantizer, FP)  |
| --initializer | initializer for quantization parameters 
| --lr | learning rate for weights 
| --coeff_qparm_lr | learning rate for quantization parameters by lr*coeff_qparm_lr
| --weight_decay | weight decay for weights 
| --qparm_wd | weight decay for quantization paramters 


## Training (PreactResnet20@Cifar-10)
0. prepare Cifar-10 dataset and place at "./data" \
prepare pretrained model from pytorchcv (https://pypi.org/project/pytorchcv/) and place at "model_zoo/pytorchcv".

1. Run nuLSQ-A as follows:

```sh
python -u main_config_preresnet20_nuLSQ_for_bash.py \
--lr 0.04 \
--coeff_qparm_lr 0.01 \
--weight_decay 1e-4 \
--qparm_wd 1e-4 \
--x_quantizer Positive_nuLSQ_quantizer \
--w_quantizer LSQ_quantizer \
--initializer NMSE_initializer \
--num_bits 2 \
--train_id 0 \
--init_from xxx.pth #[path to the initial pre-trained model] 
```

nuLSQ-WA are obtained by changing x_quantizer/w_quantizer to
```sh
--x_quantizer Positive_nuLSQ_quantizer
--w_quantizer Symmetric_nuLSQ_quantizer
```

The experiments at other datasets(cifar100/Imagenet) and models can be done by preparing yaml file under ./config and assigning the yaml file (l21). 

## References
  
During the development process, some of the code was referenced and integrated from other GitHub repositories, with proper attribution and compliance with licensing agreements.
```sh
@inproceedings{
Li2020Additive,
title={Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks},
author={Yuhang Li and Xin Dong and Wei Wang},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BkgXT24tDS}
}
```
```sh
@ARTICLE{9383003,
  author={P. {Pham} and J. A. {Abraham} and J. {Chung}},
  journal={IEEE Access}, 
  title={Training Multi-Bit Quantized and Binarized Networks with a Learnable Symmetric Quantizer}, 
  year={2021},
  volume={9},
  number={},
  pages={47194-47203},
  doi={10.1109/ACCESS.2021.3067889}}
```
```sh
@misc{pytorchcv,
  url = {https://github.com/osmr/imgclsmob},
  year = 2018
}
```
```sh
@InProceedings{pmlr-v162-nagel22a,
  title = 	 {Overcoming Oscillations in Quantization-Aware Training},
  author =       {Nagel, Markus and Fournarakis, Marios and Bondarenko, Yelysei and Blankevoort, Tijmen},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {16318--16330},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/nagel22a/nagel22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/nagel22a.html}
  }
```


