# MargNet
This is the code repository of MargNet. The necessary code for the paper is all included in this repository. 

## Introduction
We construct a new deep learning method for DP tabular data synthesis. The detailed code of MargNet is in fold `method/MargDL`


## Quick Start 
### Hyper-parameter Introduction
The code for running experiments is in main.py. The detailed description of the hyper-parameters are give as follows.
* `method`: which synthesis method you will run. `marggan` corresponds to our method MargNet
* `dataset`: name of dataset.
* `device`: the device used for running algorithms. 
* `epsilon`: DP parameter, which must be delivered when running code. 
* `--delta`: DP parameter, which is set to $1e-5$ by default.
* `--num_preprocess`: preprocessing method for numerical attributes, which is set to uniform binning by default. 
* `--rare_threshold`: threshold of preprocessing method for categorical attributes, which is set to $0.2\%$ by default.
* `--sample_device`: device used for sample data, by default is set to the same as running device.
* `--resample`: whether model use a fixed input or resampled input
* `--graph_sample`: correspond to a hybrid method, which utilizes junction tree structure to generate data from deep learning model.  


### Preparation
The necessary packages for the environment are listed in file `requirement.txt`. Firstly, make sure the datasets are put in the correct fold (we provide the Adult dataset in our repo for authors to test the code). In addition, please remember to fit the evaluation model before synthesize data. 
```
python evaluator/tune_eval_model.py adult catboost cv cuda:0
```


### Evaluation
After you activate your enviroment, try the following code to make an evaluation. 
```
python main.py marggan adult cuda:0 1.0 
```


### Ablation 
The code for ablation can be executed by `ablation.py`. For example, if you want to fit MargNet and AIM with a same marginals and compare their fitting ability, you can run the following code 
```
python ablation.py marggan adult cuda:0 10.0 --marg_num 10
python ablation.py aim adult cuda:0 10.0 --marg_num 10
```


## Results Collection
The code for evaluation is in file `evaluator/eval_seeds.py`. By default, we generate data 5 times and conduct evaluation each time we generate the data. The results are the average of all evaluations. All the results are collected in JSON format and saved in the fold `exp/{name of dataset}/{name of method}`, which can be used for further analysis. 


## Acknowledge 
We choose many baselines in our paper for evaluation. Part of the this code is from [AIM](https://github.com/ryan112358/private-pgm), [DP-MERF](https://github.com/ParkLabML/DP-MERF), [GEM](https://github.com/terranceliu/iterative-dp?tab=readme-ov-file), [PrivSyn](https://github.com/agl-c/deid2_dpsyn), [RAP++](https://github.com/amazon-science/relaxed-adaptive-projection), [TabDDPM](https://github.com/yandex-research/tab-ddpm), [CTGAN](https://github.com/juliecious/CTGAN/tree/DP). We sincerely thank them for their contribution to the community.