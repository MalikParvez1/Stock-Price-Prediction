from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from env import CryptoEnv
import pandas as pd
import os

df = pd.read_csv('ETHUSD_1.csv')


df.rename(columns={"1438956180": "Date", "3.0": "Open","3.0.1": "High", "3.0.2": "Low", "3.0.3": "Close", "81.85727776": "Volume", "2": "Trades"}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='s')

#initialization of the Environment
env = DummyVecEnv([lambda: CryptoEnv(df)])

# Instanciate the agent
model = PPO2(MlpPolicy, env, gamma=1, learning_rate=0.01, verbose=0)


