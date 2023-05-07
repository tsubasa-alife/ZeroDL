# 上書き方式の反復方策評価
# 現状の価値関数を保持しないため更新が高速になる
V = {'L1': 0.0, 'L2': 0.0}

cnt = 0
while True:
	t = 0.5 * (-1 + 0.9 * V['L1']) + 0.5 * (1 + 0.9 * V['L2'])
	delta = abs(t - V['L1'])
	V['L1'] = t

	t = 0.5 * (0 + 0.9 * V['L1']) + 0.5 * (-1 + 0.9 * V['L2'])
	delta = max(delta, abs(t - V['L2']))
	V['L2'] = t

	cnt += 1
	if delta < 0.0001:
		print(V)
		print(cnt)
		break
