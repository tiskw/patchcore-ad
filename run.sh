#!/bin/sh

python3 main_mvtecad.py runall --model resnet18
python3 main_mvtecad.py runall --model resnet34
python3 main_mvtecad.py runall --model resnet50
python3 main_mvtecad.py runall --model resnet101
python3 main_mvtecad.py runall --model resnet152
python3 main_mvtecad.py runall --model wide_resnet50_2

python3 main_mvtecad.py runall --model data_resnext50_32x4d
python3 main_mvtecad.py runall --model data_resnext101_32x8d

python3 main_mvtecad.py runall --model densenet121
python3 main_mvtecad.py runall --model densenet161
python3 main_mvtecad.py runall --model densenet169
python3 main_mvtecad.py runall --model densenet201

python3 main_mvtecad.py runall --model vgg11_bn
python3 main_mvtecad.py runall --model vgg13_bn
python3 main_mvtecad.py runall --model vgg16_bn
python3 main_mvtecad.py runall --model vgg19_bn

python3 main_mvtecad.py runall --model deeplabv3_resnet50
python3 main_mvtecad.py runall --test_only --repo facebookresearch/semi-supervised-ImageNet1K-models --model resnet50_swsl
python3 main_mvtecad.py runall --test_only --repo facebookresearch/semi-supervised-ImageNet1K-models --model resnet50_ssl
