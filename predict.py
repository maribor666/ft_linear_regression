# add argparse
thetas_path = "./thetas.txt"

milleage = 100

def main():
	thetas = open(thetas_path, mode='r').readlines()
	theta0 = int(thetas[0])
	theta1 = int(thetas[1])
	hyp = theta0 + milleage * theta1
	print(f"Calculated price should be: { hyp }")


if __name__ == '__main__':
	main()
