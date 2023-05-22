import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class Policy(Model):
	def __init__(self, action_size):
		super().__init__()
		self.l1 = L.Linear(128)
		self.l2 = L.Linear(action_size)

	def forward(self, x):
		h = F.relu(self.l1(x))
		h = self.l2(h)
		return F.softmax(h)


class Agent:
	def __init__(self):
		self.gamma = 0.98
		self.lr = 0.0002
		self.action_size = 2

		self.memory = []
		self.pi = Policy(self.action_size)
		self.optimizer = optimizers.Adam(self.lr)
		self.optimizer.setup(self.pi)

	def get_action(self, state):
		state = state[np.newaxis, :]
		probs = self.pi(state)
		probs = probs[0]
		action = np.random.choice(len(probs), p=probs.data)
		return action, probs[action]