# Deep Insights into Noisy Pseudo Labeling on Graph Data

This is a PyTorch implementation of [Deep Insights into Noisy Pseudo Labeling on Graph Data](https://arxiv.org/abs/2310.01634),
NeurIPS 2023, *Botao Wang, Jia Li, Yang Liu, Jiashun Cheng, Yu Rong, Wenjia Wang, Fugee Tsung*

Node classification
----------------------

```
python main_node.py 
```

Link prediction
----------------------

Refer to the folder "/Link" for different base models. For example, if you with to run on **GAE**, you can use

```
python Link/GAE/run_multview.py 
```

Citation
----------------------

```
@article{wang2023deepinsight,
  title={Deep Insights into Noisy Pseudo Labeling on Graph Data},
  author={Wang, Botao and Li, Jia and Liu, Yang and Cheng, Jiashun and Rong, Yu and Wang, Wenjia and Tsung, Fugee},
  journal={NeurIPS},
  year={2023}
}
```