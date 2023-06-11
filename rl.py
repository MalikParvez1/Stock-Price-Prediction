import datetime
import time
import copy
import numpy as np
import pandas as pd
import chainer
import chainer.functions as F
import chainer.links as L
from plotly.graph_objs import Candlestick, Layout
from plotly.offline import init_notebook_mode

init_notebook_mode()

data = pd.read_csv('ETHUSD_1.csv')
data.columns = data.columns.str.strip()

data.rename(columns={"1438956180": "Date", "3.0": "Open","3.0.1": "High", "3.0.2": "Low", "3.0.3": "Close", "81.85727776": "Volume", "2": "Trades"}, inplace=True)
print(data.columns)
data['Date'] = pd.to_datetime(data['Date'], unit='s')

date_split = '2020-11-02 01:40:00'
train = data[data['Date'] < date_split]
test = data[data['Date'] >= date_split]

def plot_train_test(train, test, date_split):
    data = [
        Candlestick(x=train.index, open=train['Open'], high=train['High'], low=train['Low'], close=train['Close'], name='train'),
        Candlestick(x=test.index, open=test['Open'], high=test['High'], low=test['Low'], close=test['Close'], name='test')
    ]

    layout = Layout(
        title='Stock Price',
        xaxis=dict(
            title='Date'
        ),
        yaxis=dict(
            title='Price'
        ),
        shapes=[
            dict(
                x0=date_split,
                x1=date_split,
                y0=min(train['Low'].min(), test['Low'].min()),
                y1=max(train['High'].max(), test['High'].max()),
                xref='x',
                yref='y',
                line=dict(
                    color='red',
                    width=1,
                    dash='dot'
                )
            )
        ],
        annotations=[
            {
                'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False,
                'xanchor': 'left', 'text': 'test_Data'
            }
        ]
    )

    # Plot the data
    figure = Figure(data=data, layout=layout)
    iplot(figure)

class Environment1:
    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()

    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_t)]
        return [self.position_value] + self.history

    def step(self, act):
        reward = 0

        # act = 0: nix tun, 1: kaufen, 2: verkaufen
        if act == 1:
            self.positions.append(self.data.iloc[self.t, :]['Close'])
        elif act == 2:
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                for p in self.positions:
                    profits += (self.data.iloc[self.t, :]['Close'] - p)
                reward += profits
                self.profits += profits
                self.positions = []

        self.t += 1
        self.position_value = 0
        for p in self.positions:
            self.position_value += (self.data.iloc[self.t, :]['Close'] - p)
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t, :]['Close'] - self.data.iloc[(self.t - 1), :]['Close'])

        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1

        return [self.position_value] + self.history, reward, self.done



def train_dqn(env):
    class Q_Network(chainer.Chain):
        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1=L.Linear(input_size, hidden_size),
                fc2=L.Linear(hidden_size, hidden_size),
                fc3=L.Linear(hidden_size, output_size)
            )

        def __call__(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            y = self.fc3(h)
            return y

        def reset(self):
            self.zerograds()

    Q = Q_Network(input_size=env.history_t + 1, hidden_size=100, output_size=3)
    Q_ast = copy.deepcopy(Q)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(Q)

    epoch_num = 50
    step_max = len(env.data) - 1
    memory_size = 200
    batch_size = 20
    epsilon = 1.0
    epsilon_decrease = 1e-3
    epsilon_min = 0.1
    start_reduce_epsilon = 200
    train_freq = 20
    gamma = 0.97
    show_low_freq = 5

    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []

    start = time.time()
    for epoch in range(epoch_num):

        pobs = env.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0

        while not done and step < step_max:

            # Select act
            pact = np.random.randint(3)
            if np.random.rand() > epsilon:
                pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))
                pact = np.argmax(pact.data)

            # Act
            obs, reward, done = env.step(pact)

            # Add memory
            memory.append((np.array(pobs), pact, reward, np.array(obs), done))
            if len(memory) > memory_size:
                memory.pop(0)

            # Train or update Q
            if len(memory) == memory_size:
                if total_step % train_freq == 0:
                    shuffled_memory = np.random.permutation(memory)
                    memory_idx = range(0, len(shuffled_memory), batch_size)
                    for i in memory_idx:
                        batch = shuffled_memory[i:i + batch_size]
                        b_pobs = np.stack(batch[:, 0])
                        b_pact = batch[:, 1]
                        b_reward = batch[:, 2]
                        b_obs = np.stack(batch[:, 3])
                        b_done = batch[:, 4]

                        q = Q(b_pobs)
                        maxq = np.max(Q_ast(b_obs).data, axis=1)
                        target = copy.deepcopy(q.data)
                        for j in range(batch_size):
                            target[j, b_pact[j]] = b_reward[j] + gamma * maxq[j] * (not b_done[j])
                        Q.reset()
                        loss = F.mean_squared_error(q, target)
                        total_loss += loss.data
                        loss.backward()
                        optimizer.update()

                    if total_step % update_q_freq == 0:
                        Q_ast = copy.deepcopy(Q)

                # Epsilon
                if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                    epsilon -= epsilon_decrease

                # Next step
                total_reward += reward
                pobs = obs
                step += 1
                total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss)

        if (epoch + 1) % show_low_freq == 0:
            log_reward = sum(total_rewards[((epoch + 1) - show_low_freq):]) / show_low_freq
            log_loss = sum(total_losses[((epoch + 1) - show_low_freq):]) / show_low_freq
            elapsed_time = time.time() - start
            print('\t'.join(map(str, [epoch + 1, epsilon, total_step, log_reward, log_loss, elapsed_time])))
            start = time.time()

    return Q, total_losses, total_rewards

Q, total_losses, total_rewards = train_dqn(Environment1(train))


