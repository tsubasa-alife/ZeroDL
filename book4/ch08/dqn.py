import sys
sys.path.append("../common")
import numpy as np
import matplotlib.pyplot as plt
import copy
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
from replay_buffer import ReplayBuffer


class QNet(Model):
	def __init__(self, action_size):
		super().__init__()
		self.l1 = L.Linear(128)
		self.l2 = L.Linear(128)
		self.l3 = L.Linear(action_size)

	def forward(self, x):
		h = F.relu(self.l1(x))
		h = F.relu(self.l2(h))
		h = self.l3(h)
		return h


class DQNAgent:
	def __init__(self):
		self.gamma = 0.98
		self.lr = 0.0005
		self.epsilon = 0.1
		self.buffer_size = 10000
		self.batch_size = 32
		self.action_size = 2

		self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
		self.qnet = QNet(self.action_size)
		self.qnet_target = QNet(self.action_size)
		self.optimizer = optimizers.Adam(self.lr)
		self.optimizer.setup(self.qnet)

	def sync_qnet(self):
		self.qnet_target = copy.deepcopy(self.qnet)

	def get_action(self, state):
		if np.random.rand() < self.epsilon:
			return np.random.choice(self.action_size)
		else:
			state = state[np.newaxis, :]
			qs = self.qnet(state)
			return qs.data.argmax()

	def update(self, state, action, reward, next_state, done):
		self.replay_buffer.add(state, action, reward, next_state, done)
		if len(self.replay_buffer) < self.batch_size:
			return

		state, action, reward, next_state, done = self.replay_buffer.get_batch()
		qs = self.qnet(state)
		q = qs[np.arange(self.batch_size), action]

		next_qs = self.qnet_target(next_state)
		next_q = next_qs.max(axis=1)
		next_q.unchain()
		target = reward + self.gamma * next_q * (1 - done)

		loss = F.mean_squared_error(q, target)

		self.qnet.cleargrads()
		loss.backward()
		self.optimizer.update()


episodes = 300
sync_interval = 20
env = gym.make("CartPole-v0")
agent = DQNAgent()
reward_history = []

for episode in range(episodes):
	state = env.reset()
	done = False
	total_reward = 0

	while not done:
		action = agent.get_action(state)
		next_state, reward, done, info = env.step(action)

		agent.update(state, action, reward, next_state, done)
		state = next_state
		total_reward += reward

	if episode % sync_interval == 0:
		agent.sync_qnet()

	reward_history.append(total_reward)
	if episode % 10 == 0:
		print("episode :{}, total reward : {}".format(episode, total_reward))

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(range(len(reward_history)), reward_history)
plt.show()


agent.epsilon = 0
state = env.reset()
done = False
total_reward = 0

while not done:
	action = agent.get_action(state)
	next_state, reward, done, info = env.step(action)
	state = next_state
	total_reward += reward
	env.render()

print("total reward : {}".format(total_reward))