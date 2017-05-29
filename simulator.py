import pickle
import numpy as np
import matplotlib.pyplot as plt
from user import User
from PBM_TS import PBM_TS
from RandomPick import RandomPick
from TS import TS
from PBM_BUCB import PBM_BUCB
from BUCB import BUCB

class Simulator:
	def __init__(self, itemid, posProb, itemProb):
		self.L = len(posProb)
		self.K = len(itemProb)
		self.itemid = itemid
		self.posProb = posProb
		self.itemProb = itemProb

	def save(self, fname):
		with open(fname, 'w') as f:
			pickle.dump(self, f)

	def load(self, fname):
		load_sim = pickle.load(fname)
		self.itemid = load_sim.itemid
		self.posProb = load_sim.posProb
		self.itemProb = load_sim.itemProb

	def run(self, log_fname, step_cnt=1000, save_graph=False):
		user = User(self.posProb, self.itemProb)
		models = [ 
			PBM_TS(self.itemid, self.posProb),
			#RandomPick(self.itemid, self.posProb),
			TS(self.itemid, self.posProb),
			BUCB(self.itemid, self.posProb),
			PBM_BUCB(self.itemid, self.posProb)
		]

		labels = ['PBM_TS', 'TS', 'BUCB', 'PBM_BUCB']
		regret = [0.0] * len(models)
		plots = [[] for _ in range(len(models))]
		step = 0
		pointNum = 10000
		base_step = []

		while step < step_cnt:
			step += 1
			if step % 10 == 0:
				print(step, step_cnt)

			for i in range(len(models)):
				selected_items = models[i].select_items(self.L)
				feedback = user.react(selected_items)
				models[i].update(selected_items, feedback)
				regret[i] += sum([self.posProb[l] * self.itemProb[l] for l in range(self.L)]) \
					- feedback.count(True)

			if step % max(1, (step / pointNum)) == 0:
				base_step.append(step)
				for i in range(len(models)):
					plots[i].append(regret[i])

		if save_graph:
			plt.figure()
			for i in range(len(models)):
				plt.plot(base_step, plots[i], label=labels[i])
			plt.legend()
			plt.xlabel('step')
			plt.ylabel('regret')
			plt.savefig(log_fname+'graph.png')

		return base_step, np.array(plots)
		#models[0].save(log_fname+'PBM_TS.pickle')

		#with open(log_fname, 'w') as f:
		#	f.write(str(plots[0]))
