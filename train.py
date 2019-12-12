import argparse

import numpy as np
import matplotlib.pyplot as plt

def_data_path = "./data.csv"
save_path = "./thetas.txt"
# add: specify number of iterations(iters < 1000000). - 1 bonus
# add option to plot or not                      	  - 2 bonus
# add 1 another small  data set - 				      - 1 bonus
# add specify a delimeter in dataset.                 - 1 bonus
# add data path Specify 							  - 1 bonus
 
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-iters', default=1000, help="Number of iterations to train.")
	parser.add_argument('-plot', default='n', help="Type y for plot graph, n for not.", type=str)
	parser.add_argument('-delimiter', default=',', help="Specify a delimeter. only ', ; : ' ' is allowed", type=str)
	parser.add_argument('-data_path', default=def_data_path, help="Specify your own dataset.", type=str)
	args = parser.parse_args()
	iters = args.iters
	delimiter = args.delimiter
	data_path = args.data_path

	if args.plot != 'y' and args.plot != 'n':
		print("-plot argument can be only 'y' or 'n'")
		exit()
	if delimiter not in [',', ':', ';', ' ']:
		print("This delimiter is not allowed.")
		exit()

	file = open(data_path, mode='r') # add try except for invalid data path
	try:
		x_label, y_label = file.readlines()[0].split(delimiter)
	except ValueError:
		print("Invalid delimeter in dataset.")
		exit()
	file.close()

	data = np.genfromtxt(data_path, delimiter=delimiter) 
	data = data[1:]
	try:
		x = [row[0] for row in data]
		y = [row[1] for row in data]
	except IndexError:
		print("Invalid delimiter in dataset.")
		exit()
	x_norm, y_norm = normalize(x), normalize(y)
	t0 = 1.0
	t1 = 1.0
	t0, t1 = train(x_norm, y_norm, t0, t1, iters=iters)

	save_theta(save_path, t0, t1)

	if args.plot == 'y':
		x_regr = np.arange(0, 1, 0.01)
		y_regr = [t0 + t1 * xi for xi in x_regr]
		plt.figure(num="ft_linear_regression")
		plt.plot(x_norm, y_norm, marker='h', label="Data graph")
		plt.plot(x_regr, y_regr, label="Regression graph")
		plt.grid(True)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.legend()
		plt.show()


def save_theta(save_path, t0, t1):
	file = open(save_path, mode="w+")
	file.write(str(t0) + "\n")
	file.write(str(t1) + "\n")

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
