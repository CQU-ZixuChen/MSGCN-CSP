# MSGCN-CSP
The source code of the model: Multi-Scale Graph Convolutional Network with Contrastive-Learning enhanced Self-attention Pooling (MSGCN-CSP). 
The parallel shaft gearbox dataset used in this paper is originated from https://github.com/HazeDT/PHMGNNBenchmark?tab=readme-ov-file.

![image](https://github.com/CQU-ZixuChen/MSGCN-CSP/blob/main/GA.png)

If this code is helpful to you, please cite this paper as follows, thank you!
@article{CHEN2024114497,
title = {A multi-scale graph convolutional network with contrastive-learning enhanced self-attention pooling for intelligent fault diagnosis of gearbox},
journal = {Measurement},
volume = {230},
pages = {114497},
year = {2024},
issn = {0263-2241},
doi = {https://doi.org/10.1016/j.measurement.2024.114497},
url = {https://www.sciencedirect.com/science/article/pii/S0263224124003828},
author = {Zixu Chen and Jinchen Ji and Wennian Yu and Qing Ni and Guoliang Lu and Xiaojun Chang},
keywords = {Graph convolutional network, Multi-scale graph, Contrastive-learning, Graph pooling, Intelligent fault diagnosis},
abstract = {Recently, the emerging graph convolutional network (GCN) has been applied into fault diagnosis with the aim of providing additional fault features through topological information. However, there are some limitations with these methods. First, the interactions between multi-frequency scales are ignored in existing studies, while they mainly focus on constructing graphs through the relationship between channels/instances. Second, the constructed graph cannot well reflect the topology of noisy samples and lacks robust hierarchical representation learning capability, and the learned graphs have limited interpretability. Hence, a Multi-Scale GCN with Contrastive-learning enhanced Self-attention Pooling (MSGCN-CSP) method is proposed for intelligent fault diagnosis of gearbox. Time–frequency distributions are converted into multi-scale graphs to extract fault features through topological relationships between multi-frequencies. Contrastive-learning is used to implement graph pooling, which enables hierarchical representation learning. Experimental results on two gearbox datasets illustrate that the proposed method offers competitive diagnostic performance and provides good interpretability in establishing GCN.}
}
