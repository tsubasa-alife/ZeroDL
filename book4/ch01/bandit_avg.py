import numpy as np
import matplotlib.pyplot as plt


class Bandit:
	def __init__(self, arms=10):
		self.rates = np.random.rand(arms)

	def play(self, arm):
		rate = self.rates[arm]
		if rate > np.random.rand():
			return 1
		else:
			return 0


class Agent:
	def __init__(self, epsilon, action_size=10):
		self.epsilon = epsilon
		self.Qs = np.zeros(action_size)
		self.ns = np.zeros(action_size)

	def update(self, action, reward):
		self.ns[action] += 1
		self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

	def get_action(self):
		if self.epsilon > np.random.rand():
			return np.random.randint(0, len(self.Qs))
		else:
			return np.argmax(self.Qs)


runs = 200
steps = 1000
# epsilonが大きいと探索が大きくなる
epsilon = 0.3
all_rates = np.zeros((runs, steps))
# 乱数のシードを固定すると、毎回同じ結果になる
# np.random.seed(0)

for run in range(runs):
	bandit = Bandit()
	agent = Agent(epsilon)
	total_reward = 0
	rates = []

	for step in range(steps):
		action = agent.get_action()
		reward = bandit.play(action)
		agent.update(action, reward)
		total_reward += reward
		rates.append(total_reward / (step + 1))

	all_rates[run] = rates

avg_rates = np.average(all_rates, axis=0)

plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(avg_rates)
plt.show()
