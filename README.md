# ddpg_agent
Reinforcement Learning agent using Deep Deterministic Policy Gradients (DDPG).

This reinforcement lerning model is a modified version of [Udacity's DDPG model](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) which is based on the paper [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971). This project was developed as part of the [Machine Learning Engineer Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t) quadcopter project and the model is based on code provided in the project assignment.

Solving OpenAI Gym's [MountainCarContinuous-v0](https://github.com/openai/gym/wiki/MountainCarContinuous-v0) continuous control problem with this model provides a particularly good learning example as its 2-dimensional continuous state space (position and velocity) and 1-dimensional continuous action space (forward, backward) are easy to visualize in two dimensions, lending to an intuitive understanding of hyperparameter tuning. 

Project development began as a kaggle kernel. Initial code in this repo is based on [DDPG_OpenAI-MountainCarContinuous-V0 Version 74](https://www.kaggle.com/samhiatt/mountaincarcontinuous-v0-ddpg?scriptVersionId=16052313). 

## Usage
See `Solving MountainCarContinuous-v0.ipynb` for an example of usage and a demo training visualization output. 

## Credits
* [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
* Andre Muta's [DDPG-MountainCarContinuous-v0](https://github.com/amuta/DDPG-MountainCarContinuous-v0) repo was helpful in suggesting some good visualizations as well as giving some good hyperparameters to start with. It looks like he uses the same code from the nanodegree quadcopter project and uses it to solve the MountainCarContinuous problem as well. His [plot_Q method in MountainCar.py](https://github.com/amuta/DDPG-MountainCarContinuous-v0/blob/master/MountainCar.py) was particularly helpful by showing how to plot Q_max, Q_std, Action at Q_max, and Policy. Adding a visualization of the policy gradients and animating the training process ended up helping me better understand the problem and the effects of various hypterparemeters. 
* Thanks to [Eli Bendersky](https://eli.thegreenplace.net/2016/drawing-animated-gifs-with-matplotlib/) for help with matplotlib animations. 
* Thanks to [Joseph Long](https://joseph-long.com/writing/colorbars/) for help with matplotlib colorbar axes placement.
