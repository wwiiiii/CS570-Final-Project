import os
import time
import datetime
from simulator import Simulator

def main():
	'''K = 25# number of items, 0-base
	L = 10# number of position, 0-base
	itemid = [i for i in range(K)]
	posProb = [0.5] * L # observation probability for certain position, decreasing order
	itemProb = [0.05] * K # click probability when observed for certain item
	'''
	if not os.path.exists('log'):
		os.makedirs('log')

	K = 7
	L = 3
	itemid = [i for i in range(K)]
	posProb = [0.8, 0.5, 0.3]
	itemProb = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]



	sim = Simulator(itemid, posProb, itemProb)
	sim.run(step_cnt=100000, log_fname='log/'+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.'))


if __name__ == '__main__':
	main()
