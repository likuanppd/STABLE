# STABLE

This repo is for source code of KDD 2022 paper "Reliable Representations Make A Stronger Defender:
Unsupervised Structure Refinement for Robust GNN".

Paper Link: https://arxiv.org/abs/2207.00012

## Environment

- python == 3.8.8
- pytorch == 1.8.2--cuda11.1
- scipy == 1.6.2
- numpy == 1.20.1
- deeprobust

## Prepare Datasets
First, you need to install Deeprobust. Here we only provide the code of MetaAttack. If you need other attack methods (DICE, Random), you
can refer to: https://github.com/DSE-MSU/DeepRobust/tree/master/examples/graph. Likewise, you can also prepare your own perturbed
graphs you need in any way.
```python
pip install deeprobust
```
Then, you can generate the perturbed graphs via
```python
python generate_attack.py --dataset cora --ptb_rate 0.05
```

## Main Method
```python
python main.py --dataset cora --ptb_rate 0.05 --alpha -0.3 --beta 2 --k 5 --jt 0.03 --cos 0.1
```

## Hyper-parameters
Though we have five hyper-parameters, they can be easily tuned according to the perturbation rate. 
Here we provide a guidance and the  specific values
which achieve the peak performance in our experiments.

- **alpha:** proportional to the perturbation rate
- **beta:** fixed at 2
- **k** proportional to the perturbation rate
- **jt:** tuned from 0.0 to 0.05, proportional to the perturbation rate
- **ct:** tuned from 0.1 to 0.3, mostly fixed at 0.1

### Cora

| ptb_rate | 0%   | 5%   | 10%  | 15%  | 20%  |
|----------|------|------|------|------|------|
| alpha    | -0.5 | -0.3 | 0.3  | 0.6  | 0.6  |
| beta     | 2    | 2    | 2    | 2    | 2    |
| k        | 1    | 5    | 7    | 7    | 7    |
| jt       | 0.0  | 0.03 | 0.03 | 0.03 | 0.03 |
| cos      | 0.1  | 0.1  | 0.1  | 0.2  | 0.25 |

### Citeseer

| ptb_rate | 0%   | 5%   | 10%  | 15%  | 20%  |
|----------|------|------|------|------|------|
| alpha    | -0.5 | -0.3 | -0.1 | -0.1 | 0.1  |
| beta     | 2    | 2    | 2    | 2    | 2    |
| k        | 3    | 3    | 5    | 5    | 5    |
| jt       | 0.0  | 0.02 | 0.02 | 0.04 | 0.03 |
| cos      | 0.1  | 0.1  | 0.1  | 0.1  | 0.1  |

### Polblogs

| ptb_rate | 0%   | 5%  | 10% | 15% | 20% |
|----------|------|-----|-----|-----|-----|
| alpha    | -0.5 | 0.3 | 0.5 | 2   | 2   |
| beta     | 2    | 1   | 1   | 2   | 2   |
| k        | 0    | 3   | 3   | 3   | 3   |
| jt       | /    | /   | /   | /   | /   |
| cos      | 0.1  | 0.1 | 0.1 | 0.1 | 0.1 |

## Citation
```
@inproceedings{li2022reliable,
  title={Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN},
  author={Li, Kuan and Liu, Yang and Ao, Xiang and Chi, Jianfeng and Feng, Jinghua and Yang, Hao and He, Qing},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={925--935},
  year={2022}
}
```

## Contact

If you have any questions, please feel free to contact me with [likuan20s@ict.ac.cn](mailto:likuan20s@ict.ac.cn).

