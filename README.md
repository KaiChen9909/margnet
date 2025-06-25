# MargNet
This is the code repository of MargNet. The necessary code for the paper is all included in this repository. 

## Introduction
We construct a new deep learning method for DP tabular data synthesis.


## Quick Start 
### Hyper-parameter Introduction
The code for running experiments is in main.py. The detailed description of the hyper-parameters are give as follows.
* `method`: which synthesis method you will run.
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
The necessary packages for the environment are listed in file `requirement.txt`. Firstly, make sure the datasets are put in the correct fold. In this repository, the evaluation model is already tuned so users do not need any operation. Otherwise, you should tune the evaluation model (using the following code) before any further operation.
```
python evaluator/tune_eval_model.py bank mlp cv cuda:0
```


### Overall Evaluation
After you activate your enviroment, try the following code to make an overall evaluation. In our paper, we by default set `num_preprocess` to be "uniform_kbins" except for DP-MERF and TabDDPM, and set `rare_threshold` to 0.002 for overall evaluation. 
```
python main.py aim bank cuda:0 1.0 --num_preprocess uniform_kbins --rare_threshold 0.002
```


### Preprocessing Investigation
If you want to try other preprocessing methods or preprocessing hyper-parameter settings, you can modify the value of preprocessing hyper-parameters like this 
```
python main.py aim bank cuda:0 1.0 --num_preprocess privtree --rare_threshold 0.01
```


### Module Comparison 
In the experiment section of the paper, we compare different modules by comparing the performances of different reconstructed algorithms.
These reconstructed algorithms are allocated new names, which can be delivered to `method`. 
For example, if you want to try PrivSyn selector with generative network synthesizer, you can try
```
python main.py gem_syn bank cuda:0 1.0 --num_preprocess uniform_kbins --rare_threshold 0.002
```




## Results Collection
The code for evaluation is in file `evaluator/eval_seeds.py`. By default, we generate data 5 times and conduct evaluation each time we generate the data. The results are the average of all evaluations. All the results are collected in JSON format and saved in the fold `exp/{name of dataset}/{name of method}`, which can be used for further analysis. 


## Acknowledge 
Part of the code is from [AIM](https://github.com/ryan112358/private-pgm), [DP-MERF](https://github.com/ParkLabML/DP-MERF), [GEM](https://github.com/terranceliu/iterative-dp?tab=readme-ov-file), [PrivSyn](https://github.com/agl-c/deid2_dpsyn), [RAP++](https://github.com/amazon-science/relaxed-adaptive-projection), [TabDDPM](https://github.com/yandex-research/tab-ddpm). We sincerely thank them for their contribution to the community.