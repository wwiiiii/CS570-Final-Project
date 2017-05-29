import numpy as np
import scipy as sp
import pickle
from scipy.stats import beta

#based on Bayes-UCB

class PBM_BUCB:
	def __init__(self, itemid, posProb):
		self.K = len(itemid)
		self.L = len(posProb)
		self.turn = 0 # how many times selected item list
		self.itemid = itemid
		self.posProb = posProb
		self.S = [[1 for _ in range(self.L)] for __ in range(self.K)]
		self.N = [[2 for _ in range(self.L)] for __ in range(self.K)]
		self.Nc = [[2.0 for _ in range(self.L)] for __ in range(self.K)]

	
	#return sorted list of item numbers, with length require_num
	def select_items(self, required_num):
		self.turn += 1
		sample_val = [0.0] * self.K
		for k in range(self.K):
			z0 = beta.ppf(1.0 - 1.0 / self.turn, sum(self.S[k]), max(0.001, sum(self.Nc[k]) - sum(self.S[k])))
			sample_val[k] = z0
		items = sorted([(sample_val[k], k) for k in range(self.K)], reverse=True)
		result = [items[i][1] for i in range(required_num)]
		return result


	def update(self, selected_items, feedback):
		assert len(selected_items) == len(feedback)
		for l in range(len(selected_items)):
			k = selected_items[l]
			self.N[k][l] += 1
			self.Nc[k][l] += self.posProb[l]
			if feedback[l]:
				self.S[k][l] += 1
	
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
			self.Nc = sim.Nc