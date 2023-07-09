from Reinforcement_Learning.gymtest import CryptoEnv
import pandas as pd
import numpy as np



from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.logger import configure

from finrl.config import TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG

df = pd.read_csv('../ETHUSD_1.csv')

check_and_make_directories([TRAINED_MODEL_DIR])


df.rename(columns={"1438956180": "Date", "3.0": "Open","3.0.1": "High", "3.0.2": "Low", "3.0.3": "Close", "81.85727776": "Volume", "2": "Trades"}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='s')
df['tic'] = 'ETHUSD'

#initialization of the Environment
env = DummyVecEnv([lambda: CryptoEnv(df)])

#TRAIN TEST DATA
date_split = '2020-11-02 01:40:00'
train = df[df['Date'] < date_split]
test = df[df['Date'] >= date_split]

env_train = DummyVecEnv([lambda: CryptoEnv(train)])

agent = DRLAgent(env = env_train)

# Set the corresponding values to 'True' for the algorithms that you want to use
if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True

model_a2c = agent.get_model("a2c")

if if_using_a2c:
  # set up logger
  tmp_path = RESULTS_DIR + '/a2c'
  new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_a2c.set_logger(new_logger_a2c)



n_actions = env.action_space.shape[-1]

action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
model.save("ddpg_pendulum")
vec_env = model.get_env()
