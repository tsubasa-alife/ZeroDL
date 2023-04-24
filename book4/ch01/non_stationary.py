import numpy as np
import matplotlib.pyplot as plt


class NonStatBandit:
	def __init__(self, arms=10):
		self.arms = arms
		self.rates = np.random.rand(arms)

	def play(self, arm):
		rate = self.rates[arm]
		self.rates += 0.1 * np.random.randn(self.arms)
		if rate > np.random.rand():
			return 1
		else:
			return 0


class AlphaAgent:
	def __init__(self, epsilon, alpha, actions=10):
		self.epsilon = epsilon
		self.alpha = alpha
		self.Qs = np.zeros(actions)

	def update(self, action, reward):
		self.Qs[action] += self.alpha * (reward - self.Qs[action])

	def get_action(self):
		if self.epsilon > np.random.rand():
			return np.random.randint(0, len(self.Qs))
		else:
			return np.argmax(self.Qs)


runs = 200
steps = 1000
# epsilonが大きいと探索が大きくなる
epsilon = 0.3
# alphaが大きいと過去の報酬の重みが大きくなる
alpha = 0.8
all_rates = np.zeros((runs, steps))
# 乱数のシードを固定すると、毎回同じ結果になる
# np.random.seed(0)

for run in range(runs):
	bandit = NonStatBandit()
	agent = AlphaAgent(epsilon, alpha)
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