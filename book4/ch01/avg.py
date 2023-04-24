import numpy as np
np.random.seed(0)
rewards = []
for n in range(1,11):
	reward = np.random.rand()
	rewards.append(reward)
	Q = sum(rewards) / n
	print(Q)

print("逐次方式")
Q = 0

# range内の数字を大きくするとnが大きいときにQ値がほとんど変化しなくなる様子がみられる
for n in range(1,1100):
	reward = np.random.rand()
	Q = Q + (reward - Q) / n
	print(Q)