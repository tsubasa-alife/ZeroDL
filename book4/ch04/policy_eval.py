import sys
sys.path.append("../common")
from collections import defaultdict
from gridworld import GridWorld

def eval_onestep(pi, V, env, gammma=0.9):
	for state in env.states():
		if state == env.goal_state:
			V[state] = 0
			continue

		action_probs = pi[state]
		new_V = 0

		# 行動の確率分布を取り出しV値を計算
		for action, action_prob in action_probs.items():
			next_state = env.next_state(state, action)
			r = env.reward(state, action, next_state)
			new_V += action_prob * (r + gammma * V[next_state])
		V[state] = new_V

	return V

def policy_eval(pi, V, env, gammma, threshold=1e-3):
	while True:
		old_V = V.copy()
		V = eval_onestep(pi, V, env, gammma)
		delta = 0
		for state in V.keys():
			# V値の変化量を計算
			t = abs(old_V[state] - V[state])
			if t > delta:
				delta = t

		if delta < threshold:
			break

	return V

env = GridWorld()
gamma = 0.9
# 0: 上, 1: 下, 2: 左, 3: 右
# これらの行動確率の分布を変更するとV値が変化する
pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 1.25})
V = defaultdict(lambda: 0)

V = policy_eval(pi, V, env, gamma)
env.render_v(V)