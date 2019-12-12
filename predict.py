import argparse

import numpy as np


def_data_path = "./data.csv"
thetas_path = "./thetas.txt"


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('milleage', type=float, nargs=1)
	parser.add_argument('-data_path', default=def_data_path, help="Specify your own dataset.", type=str)
	parser.add_argument('-delimiter', default=',', help="Specify a delimeter. only ', ; : ' ' is allowed", type=str)
	args = parser.parse_args()
	milleage = args.milleage[0]
	data_path = args.data_path
	delimiter = args.delimiter

	thetas = open(thetas_path, mode='r').readlines()
	t0 = float(thetas[0])
	t1 = float(thetas[1])

	data = np.genfromtxt(data_path, delimiter=delimiter)
	data = data[1:]
	try:
		x = [row[0] for row in data]
		y = [row[1] for row in data]
	except IndexError:
		print("Invalid delimiter in dataset.")
		exit()

	predict_price = t0 + t1 * normalize_elem(x, milleage)
	predict_price = denormalize_elem(y, predict_price)

	print(f"Calculated price is: { round(predict_price, 2) }")


def denormalize_elem(vals, to_denorm):
	return to_denorm * (max(vals) - min(vals)) + min(vals)

def normalize_elem(vals, to_norm):
	return (to_norm - min(vals)) / (max(vals - min(vals)))

if __name__ == '__main__':
	main()
