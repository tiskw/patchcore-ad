#!/bin/sh

python3 run_experiments.py run --model resnet18
python3 run_experiments.py run --model resnet34
python3 run_experiments.py run --model resnet50
python3 run_experiments.py run --model resnet101
python3 run_experiments.py run --model resnet152
python3 run_experiments.py run --model wide_resnet50_2

python3 run_experiments.py run --model data_resnext50_32x4d
python3 run_experiments.py run --model data_resnext101_32x8d

python3 run_experiments.py run --model densenet121
python3 run_experiments.py run --model densenet161
python3 run_experiments.py run --model densenet169
python3 run_experiments.py run --model densenet201

python3 run_experiments.py run --model vgg11_bn
python3 run_experiments.py run --model vgg13_bn
python3 run_experiments.py run --model vgg16_bn
python3 run_experiments.py run --model vgg19_bn

python3 run_experiments.py run --model deeplabv3_resnet50
python3 run_experiments.py run --test_only --repo facebookresearch/semi-supervised-ImageNet1K-models --model resnet50_swsl
python3 run_experiments.py run --test_only --repo facebookresearch/semi-supervised-ImageNet1K-models --model resnet50_ssl
