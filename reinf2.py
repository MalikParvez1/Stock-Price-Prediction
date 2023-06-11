import gym
import gym_anytrading

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

import numpy as np
import pandas as pd
from matplotlib import  pyplot as plt

data = pd.read_csv('ETHUSD_1.csv')