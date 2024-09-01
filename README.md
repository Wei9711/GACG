# GACG: Group-Aware Coordination Graph for Multi-Agent Reinforcement Learning

This work [Group-Aware Coordination Graph for Multi-Agent Reinforcement Learning](https://www.ijcai.org/proceedings/2024/434) appeared at the 33rd International Joint
Conference on Artificial Intelligence (IJCAI 2024)

This codebase is based on [PyMARL](https://github.com/oxwhirl/pymarl) and contains the implementation
of the [GACG](https://arxiv.org/abs/2404.10976) algorithm.

<figure>
  <img src="https://github.com/Wei9711/GACG/raw/main/GroupAwareCG.svg" alt="GACG Framework SVG">
  <figcaption> The framework of our method. GACG is designed to calculate cooperation needs between agent pairs based on current observations and to capture group-level dependencies from behaviour patterns observed across trajectories. All edges in the coordination graph as a Gaussian distribution. This graph helps agents exchange knowledge when making decisions.  During agent training, the group distance loss regularizes behaviour among agents with similar observation trajectories.
</figcaption>
</figure>

## Run an experiment 

Tasks can be found in `src/envs`. 

To run experiments on SMAC benchmark:
```shell
python src/main.py --config=gacg --env-config=sc2 with env_args.map_name='10m_vs_11m' 
```

The requirements.txt file can be used to install the necessary packages into a virtual environment.

## Baselines used in this paper
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**DCG**: Deep Coordination Graphs](https://arxiv.org/abs/1910.00091)
- [**DICG**: Deep Implicit Coordination Graphs for Multi-agent Reinforcement Learning](https://arxiv.org/abs/2006.11438) 
- [**CASEC**: Context-Aware Sparse Deep Coordination Graphs](https://arxiv.org/abs/2106.02886)
- [**VAST**: VAST: Value Function Factorization with Variable Agent Sub-Teams](https://proceedings.neurips.cc/paper_files/paper/2021/hash/c97e7a5153badb6576d8939469f58336-Abstract.html)


## Citing GACG 

If you use GACG  in your research, please cite the [GACG](https://arxiv.org/abs/2404.10976).

*Wei Duan, Jie Lu, Junyu Xuan. Group-Aware Coordination Graph for Multi-Agent Reinforcement Learning. CoRR abs/2404.10976 (2024)*

In BibTeX format:

```tex
@inproceedings{ijcai2024p434,
  title     = {Group-Aware Coordination Graph for Multi-Agent Reinforcement Learning},
  author    = {Duan, Wei and Lu, Jie and Xuan, Junyu},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {3926--3934},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/434},
  url       = {https://doi.org/10.24963/ijcai.2024/434},
}
