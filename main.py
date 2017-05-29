import os
import time
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
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

	'''K = 7
	L = 3
	itemid = [i for i in range(K)]
	posProb = [0.8, 0.5, 0.3]
	itemProb = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]'''

	K = 5
	L = 3
	itemid = [i for i in range(K)]
	posProb = [0.9, 0.6, 0.3]
	itemProb = [0.45, 0.35, 0.25, 0.15, 0.05]


	expCount = 10
	stepCount = 10000
	modelNumber = 4
	sim = Simulator(itemid, posProb, itemProb)
	log_fname='log/'+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.')

	base, plots = sim.run(step_cnt=stepCount, log_fname=log_fname)

	for i in range(expCount-1):
		print(str(i+2)+'th exp')
		base, nowp = sim.run(step_cnt=stepCount, log_fname=log_fname)
		plots += nowp

	plots = plots / float(expCount)
	labels = ['PBM_TS', 'TS', 'BUCB', 'PBM_BUCB']
	plt.figure()
	for i in range(modelNumber):
		plt.plot(base, list(plots[i]), label=labels[i])
	plt.legend()
	plt.xlabel('step')
	plt.ylabel('regret')
	plt.savefig(log_fname+'graph.png')
	with open(log_fname+'txt', 'w') as f:
		f.write(str(('labels', labels, '# of arms', K, '# of pos', L, 'posProb', posProb, 'itemProb', itemProb, 'expCnt', expCount, 'stepCnt', stepCount)))

	with open(log_fname+'.plot.pickle', 'wb') as f:
		pickle.dump(plots, f)


if __name__ == '__main__':
	main()
