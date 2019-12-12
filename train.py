from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

data_path = "./data.csv"

def main():
	data = np.genfromtxt(data_path, delimiter=',')
	data = data[1:]
	x = [row[0] for row in data]
	y = [row[1] for row in data]
	x_norm, y_norm = normalize(x), normalize(y)
	t0 = 1.0
	t1 = 1.0
	t0, t1 = train(x_norm, y_norm, t0, t1)
	print(t0, t1)

	test = t0 + t1 * normalize_elem(x, 63060)
	test = denormalize_elem(y, test)

	
def denormalize_elem(vals, to_denorm):
	return to_denorm * (max(vals) - min(vals)) + min(vals)


def normalize_elem(vals, to_norm):
	return (to_norm - min(vals)) / (max(vals - min(vals)))


def train(x, y, t0, t1, lr=0.1, iters=1000):
	m = len(x)
	error = loss(x, y, t0, t1, m)
	for n in range(iters):
		t0, t1 = update_theta(x, y, t0, t1, m, lr)
		error = loss(x, y, t0, t1, m)
	return t0, t1

def update_theta(x, y, t0, t1, m, lr):
	summa0 = 0
	summa1 = 0
	for xi, yi in zip(x, y):
		summa0 += hyp(xi, t0, t1) - yi
		summa1 += (hyp(xi, t0, t1) - yi) * xi
	t0 -= lr * (summa0 / m)
	t1 -= lr * (summa1 / m)
	return t0, t1

def loss(x, y, t0, t1, m):
	summa = 0
	for xi, yi in zip(x, y):
		summa += (hyp(xi, t0, t1) - yi) ** 2
	return summa / (m * 2)


def hyp(x, t0, t1):
	return t0 + t1 * x

def normalize(vals):
	res = []
	min_val = min(vals)
	max_val = max(vals)
	for val in vals:
		new_val = (val - min_val) / (max_val - min_val)
		res.append(new_val)
	return res


if __name__ == '__main__':
	main()
