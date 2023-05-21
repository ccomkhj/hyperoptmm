# Template using hyperOpt and Mlflow with mm-series computer vision

## MLflow
An open source platform for the end-to-end machine learning lifecycle.
You can setup your own repository and dashboard and keep tracking the performance and metrics of different models

## Hyperopt
HyperOpt is an open-source Python library for Bayesian optimization developed by James Bergstra. It is designed for large-scale optimization for models with hundreds of parameters and allows the optimization procedure to be scaled across multiple cores and multiple machines.

## OpenMMLab
OpenMMLab builds the most influential open-source computer vision algorithm system in the deep learning era. 

### This repo I give a simple demo how to use MLflow & Hyperopt & MMpretrain. So, you can build your MLops pipeline solely using opensource.

- Step 1.
Setup custom [MLflowHook](https://github.com/ccomkhj/MlflowLoggerhook)

- Step 2.
Clone this repo on top of your MMpretrain repo (it can be any MM-series).
```bash
git clone https://github.com/ccomkhj/hyperoptmm.git hyperoptmm
mv hyperoptmm path/to/mmpretrain
rm -rf hyperoptmm
```

- Step 3.
Write down relevant environments onto run_train.sh

- Step 4. 
Setup your config file.

- Step 5.
`sh run_train.sh`

In this repo, optimizer is tuned. You can setup anything you like.