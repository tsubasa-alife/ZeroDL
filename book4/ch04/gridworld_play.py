import sys
sys.path.append("../common")
import numpy as np
from gridworld import GridWorld


env = GridWorld()
V = {}
for state in env.states():
	V[state] = np.random.randn()
env.render_v(V)

