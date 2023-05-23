import numpy as np
import matplotlib.pyplot as plt
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class PolicyNet(Model):
	def __init__(self, action_size):
		super().__init__()
		self.l1 = L.Linear(128)
		self.l2 = L.Linear(action_size)

	def forward(self, x):
		h = F.relu(self.l1(x))
		h = self.l2(h)
		return F.softmax(h)


class ValueNet(Model):
	def __init__(self):
		super().__init__()
		self.l1 = L.Linear(128)
		self.l2 = L.Linear(1)

	def forward(self, x):
		h = F.relu(self.l1(x))
		h = self.l2(h)
		return h


class Agent:
	def __init__(self):
		self.gamma = 0.98
		self.lr_pi = 0.0002
		self.lr_v = 0.0005
		self.action_size = 2

		self.pi = PolicyNet(self.action_size)
		self.v = ValueNet()
		self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
		self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)

	def get_action(self, state):
		state = state[np.newaxis, :]
		probs = self.pi(state)
		probs = probs[0]
		action = np.random.choice(len(probs), p=probs.data)
		return action, probs[action]

	def update(self, state, action_prob, reward, next_state, done):
		state = state[np.newaxis, :]
		next_state = next_state[np.newaxis, :]

		target = reward + (1 - done) * self.gamma * self.v(next_state)
		target.unchain()
		v = self.v(state)
		loss_v = F.mean_squared_error(target, v)

		delta = target - v
		delta.unchain()
		loss_pi = -F.log(action_prob) * delta

		self.pi.cleargrads()
		loss_pi.backward()
		self.optimizer_pi.update()

		self.v.cleargrads()
		loss_v.backward()
		self.optimizer_v.update()


episodes = 3000
env = gym.make('CartPole-v0')
agent = Agent()
reward_history = []

for episode in range(episodes):
	state = env.reset()
	done = False
	total_reward = 0

	while not done:
		action, prob = agent.get_action(state)
		next_state, reward, done, _ = env.step(action)

		agent.update(state, prob, reward, next_state, done)
		state = next_state
		total_reward += reward

	reward_history.append(total_reward)
	if episode % 10 == 0:
		print("episode :{}, total reward : {}".format(episode, total_reward))

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(range(len(reward_history)), reward_history)
plt.show()
