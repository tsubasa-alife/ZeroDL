import sys
sys.path.append("../common")
import numpy as np
from collections import defaultdict
from gridworld import GridWorld
from utils import greedy_probs


class McAgent:
	def __init__(self):
		self.gamma = 0.9
		self.epsilon = 0.1
		self.alpha = 0.1
		self.action_size = 4

		random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
		self.pi = defaultdict(lambda: random_actions)
		self.Q = defaultdict(lambda: 0)
		self.cnts = defaultdict(lambda: 0)
		self.memory = []

	def get_action(self, state):
		action_probs = self.pi[state]
		actions = list(action_probs.keys())
		probs = list(action_probs.values())
		return np.random.choice(actions, p=probs)

	def add(self, state, action, reward):
		data = (state, action, reward)
		self.memory.append(data)

	def reset(self):
		self.memory.clear()

	def update(self):
		G = 0
		for data in reversed(self.memory):
			state, action, reward = data
			G = self.gamma * G + reward
			key = (state, action)
			self.Q[key] += (G - self.Q[key]) * self.alpha
			self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = McAgent()

episodes = 10000
for episode in range(episodes):
	state = env.reset()
	agent.reset()

	while True:
		action = agent.get_action(state)
		next_state, reward, done = env.step(action)
		agent.add(state, action, reward)

		if done:
			agent.update()
			break

		state = next_state

env.render_q(agent.Q)