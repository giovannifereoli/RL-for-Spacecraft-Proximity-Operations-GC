<p align="center">
  <img align="center" src="https://getvectorlogo.com/wp-content/uploads/2019/10/politecnico-di-milano-vector-logo.png" width="250" />
  <img align="center" src="https://www.colorado.edu/brand/sites/default/files/styles/medium/public/page/boulder-one-line-reverse.png?itok=jWuueUXe" width="400" />
</p>

<div align="center">
  
![GitHub Repo stars](https://img.shields.io/github/stars/giovannifereoli/thesis?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/giovannifereoli/ThesisVer2)
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

</div>

## Meta-Reinforcement Learning for Spacecraft Proximity Operations Guidance and Control in Cislunar Space 

In order to tackle the challenges of the future space exploration, new lightweight and model-free
guidance algorithms are needed to make spacecrafts autonomous. Indeed, in the last few decades
autonomous spacecraft guidance has become an active research topic and certainly in the next years
this technology will be needed to ensure proximity operation capabilities in the cislunar space. For
instance, NASA’s Artemis program plans to establish a lunar Gateway and this type of autonomous
manoeuvres, besides nominal rendezvous and docking (RV&D) ones, will be needed also for assembly
and maintenance procedures.

In this context a Meta-Reinforcement Learning (Meta-RL) algorithm will be applied to address the
real-time relative optimal guidance problem of a spacecraft in cislunar environment. Non-Keplerian
orbits have a more complex dynamics and classic control theory is less flexible and more
computationally expensive with respect to Machine Learning (ML) methods. Moreover, Meta-RL is
chosen for its elegant and promising ability of ‘‘learning how to learn’’ through experience.

A stochastic optimal control problem will be modelled in the Circular Restricted Three-Body Problem
(CRTBP) framework as a time-discrete Markov Decision Process (MDP). Then a Deep-RL agent,
composed by Long Short-Term Memory (LSTM) as Recurrent Neural Network (RNN), will be trained with
a state-of-the-art actor-critic algorithm known as Proximal Policy Optimization (PPO). In addition,
operational constraints and stochastic effects will be considered to assess solution safety and
robustness.

## Credits
This project has been created by [Giovanni Fereoli](https://github.com/giovannifereoli) in 2023.
For any problem, clarification or suggestion, you can contact the author at [giovanni.fereoli@mail.polimi.it](mailto:giovanni.fereoli@mail.polimi.it).

## License
The package is under the [MIT](https://choosealicense.com/licenses/mit/) license.

