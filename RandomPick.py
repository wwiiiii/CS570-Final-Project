import random

class RandomPick:
	def __init__(self, itemid, posProb):
		self.K = len(itemid)
		self.L = len(posProb)
		self.turn = 0 # how many times selected item list
		self.itemid = itemid
		self.posProb = posProb

	#return sorted list of item numbers, with length require_num
	def select_items(self, required_num):
		random.shuffle(self.itemid)
		return self.itemid[:required_num]
		

	def update(self, selected_items, feedback):
		None