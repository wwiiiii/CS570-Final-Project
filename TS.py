import numpy as np
import scipy as sp
import pickle
from scipy.stats import beta

class TS:
	def __init__(self, itemid, posProb):
		self.K = len(itemid)
		self.L = len(posProb)
		self.turn = 0 # how many times selected item list
		self.itemid = itemid
		self.posProb = posProb
		self.S = [0 for __ in range(self.K)]
		self.N = [0 for __ in range(self.K)]

	#return sorted list of item numbers, with length require_num
	def select_items(self, required_num):
		self.turn += 1
		sample_val = [0.0] * self.K
		for k in range(self.K):
			sample_val[k] = np.random.beta(self.S[k] + 1, self.N[k] - self.S[k] + 1)
		items = sorted([(sample_val[k], k) for k in range(self.K)], reverse=True)
		result = [items[i][1] for i in range(required_num)]
		return result


	def update(self, selected_items, feedback):
		assert len(selected_items) == len(feedback)
		for l in range(len(selected_items)):
			k = selected_items[l]
			self.N[k] += 1
			if feedback[l]:
				self.S[k] += 1
	
	def save(self, fname):
		with open(fname, 'wb') as f:
			pickle.dump(self, f)

	def load(self, fname):
		with open(fname, 'rb') as f:
			sim = pickle.load(f)
			self.K = sim.K
			self.L = sim.L
			self.turn = sim.turn
			self.itemid = sim.itemid
			self.posProb = sim.posProb
			self.S = sim.S
			self.N = sim.N