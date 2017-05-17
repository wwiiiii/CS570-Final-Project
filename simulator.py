import pickle
import matplotlib.pyplot as plt
from user import User
from PBM_TS import PBM_TS
from RandomPick import RandomPick

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

	def run(self, log_fname, step_cnt=1000):
		user = User(self.posProb, self.itemProb)
		pbm_ts = PBM_TS(self.itemid, self.posProb)
		rp = RandomPick(self.itemid, self.posProb)
		pt_regret = 0.0
		pt_plot = []
		rp_regret = 0.0
		rp_plot = []
		step = 0
		pointNum = 10000
		base_step = []

		while step < step_cnt:
			step += 1
			if step % 100 == 0:
				print(step, step_cnt)
			selected_items = pbm_ts.select_items(self.L)
			feedback = user.react(selected_items)
			pbm_ts.update(selected_items, feedback)
			pt_regret += sum([self.posProb[l] * self.itemProb[l] for l in range(self.L)]) \
					- feedback.count(True)#- sum([posProb[l] if feedback[l] else 0.0 for l in range(self.L)])
			
			selected_items = rp.select_items(self.L)
			feedback = user.react(selected_items)
			rp.update(selected_items, feedback)
			rp_regret += sum([self.posProb[l] * self.itemProb[l] for l in range(self.L)]) \
					- feedback.count(True)#- sum([posProb[l] if feedback[l] else 0.0 for l in range(self.L)])

			if step % (step / pointNum) == 0:
				base_step.append(step)
				pt_plot.append(pt_regret)
				rp_plot.append(rp_regret)

		plt.figure()
		plt.plot(base_step, pt_plot, label='PBM-TS')
		plt.plot(base_step, rp_plot, label='RANDOM')
		plt.legend()
		plt.xlabel('step')
		plt.ylabel('regret')
		plt.savefig(log_fname+'graph.png')
		pbm_ts.save(log_fname+'PBM_TS.pickle')

		with open(log_fname, 'w') as f:
			f.write(str(pt_plot))
