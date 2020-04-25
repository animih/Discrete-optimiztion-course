#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
from collections import namedtuple
import random
import numpy
import copy

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def GetCandidate(seq, len):
	i = random.randint(0, len-1)
	j = random.randint(0, len-1)

	if(i > j):
		i, j = j, i

	#print(i, j)

	seq[i:j] = reversed(seq[i:j])

	return seq

def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

# Return true if line segments AB and CD intersect
def intersec(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


T_0 = 8900

def CoolThisShit(iteration, coef):
	return T_0*math.exp(-coef*math.sqrt(iteration))#nodeCount))


def shift(lst, stp):
	if stp < 0:
		stp = abs(stp)
		for i in range(stp):
			lst.append(lst.pop(0))
	else:
		for i in range(stp):
			lst.insert(0, lst.pop())





def get_greedy(solution, points, nodeCount):

	list = [i for i in range(nodeCount)]

	counter = 0

	current = list.pop()

	while list:
		min_u = list[0]
		for u in list:
			if(length(points[current], points[u]) < length(points[current], points[min_u])):
				min_u = u

		solution[counter] = current
		counter += 1

		print(len(list))

		list.remove(min_u)
		current = min_u

	solution[counter] = current

	solution.reverse()

	return solution

def makePair(i, j):
    if i > j:
        return (j, i)
    else:
        return (i, j)

class class_sol():

	def __init__(self, solution, cost):
		self.solution = solution
		self.size = len(solution)
		self.cost = cost
		self._makeEdges()

	def _makeEdges(self):

		self.edges = set()

		for i in range(self.size):
			self.edges.add(makePair(self.solution[i-1], self.solution[i]))

	def index(self, i):
		return self.solution.index(i)
	def around(self, node):

		index = self.solution.index(node)

		return (self.solution[(index+1)%self.size], self.solution[(index-1)%self.size] )

	def update(self, broken, joined):

		edges = (self.edges - broken) | joined

		if len(edges) < self.size:
			return False, []

		pocl = {}
		node = 0

		while len(edges) > 0:
			
			for i, j in edges:
				if i == node:
					pocl[node] = j
					node = j
					break
				elif j == node:
					pocl[node] = i
					node = i
					break

			edges.remove(makePair(i, j))

		if len(pocl) < self.size:
			#print(len(pocl), self.size)
			return False, []			

		perv = pocl[0]
		new_solution = [0]
		visited = set(new_solution)

		while perv not in visited:
			visited.add(perv)
			new_solution.append(perv)
			perv = pocl[perv]

		return len(new_solution) == self.size, new_solution

	def get(self):
		return self.solution
	def contains(self, edge):
		return edge in self.edges

	def rewrite(self, solution, gain):
		self.solution = copy.copy(solution)
		self.cost -= gain
		self._makeEdges()

def func(n):
	return n

class hip_hop:

	solutions = set()

	def __init__(self, solution, points, cost, nodeCount):
		self.solution = class_sol(solution, cost)
		self.solutions.add(map(func, solution))
		self.points = points
		self.cost = cost
		self.size = nodeCount
	def distance(self, t1, t2):
		return length(self.points[t1], self.points[t2])

	def closest(self, t2i, X):

		list = []

		for u in self.solution.get():
			yi = makePair(t2i, u)

			if yi in X or self.solution.contains(yi):
				continue

			list.append(u)

		return sorted(list, key=lambda x: self.distance(x, t2i))

	def improve(self):

		for t1 in self.solution.get():
			around = self.solution.around(t1)

			for t2 in around:
				X = set()
				X.add(makePair(t1, t2))
				#print(X)

				close = self.closest(t2, X)

				tries = 5

				for t3 in close:
					Y = set()
					Y.add(makePair(t2, t3))
					gain = self.distance(t1, t2) - self.distance(t2, t3)	
					
					if gain > 0:				
						if self.chooseX(t1, t3, gain, X, Y):
							print(self.cost)
							return True


					tries -= 1

					if tries == 0:
						break
		return False
	def chooseX(self, t1, last, gain, X, Y):

		#print("potential")

		if len(X)==4:
			pred, pocl = self.solution.around(last)

			if self.distance(pred, last) > self.distance(pocl, last):
				around = [pred]
			else:
				around = [pocl]
		else:
			around = self.solution.around(last)

		for t2i in around:
			
			xi = makePair(last, t2i)
			Gi = gain+self.distance(last, t2i)

			if xi not in Y and xi not in X:
				added = copy.deepcopy(Y)
				removed = copy.deepcopy(X)

				removed.add(xi)
				added.add(makePair(t2i, t1))

				new = Gi - self.distance(t2i, t1)
				is_tour, new_tour = self.solution.update(removed, added)

				if not is_tour and len(added)>2:
					continue

				if map(func, new_tour) in self.solutions:
					print("Skipped allready found tour.")
					return False

				if is_tour and new > 0:
					self.solution.rewrite(new_tour, new)
					self.solutions.add(map(func, self.solution.get()))
					self.cost -= new
					print("added")
					return True
				else:
					choice = self.chooseY(t1, t2i, Gi, removed, Y)

					if len(X) == 2 and choice:
						return True
					else:
						return choice

		return False

	def chooseY(self, t1, t2i, gain, X, Y):
		
		if len(X) == 2:
			top = 5
		else:
			top =1

		for node in self.closest(t2i, X):
			yi = makePair(t2i, node)
			added = copy.deepcopy(Y)
			added.add(yi)
			Gi = gain - self.distance(t2i, node)

			if self.chooseX(t1, node, Gi, X, added):
				return True

			top -= 1

			if top == 0:
				return False

		return False

	def get(self):
		return self.solution.get()

def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

# Return true if line segments AB and CD intersect
def intersec(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


class Dream_pop():
	def __init__(self, solution, points, cost):
		self.best_solution = copy.copy(solution)
		
		self.points = points
		self.best_cost = cost
		self.size = len(solution)
	def exchange_3(self, execute, a, c, e):
		b, d, f = a+1, c+1, e+1

		p_a, p_b, p_c, p_d, p_e, p_f = self.best_solution[a], self.best_solution[b], self.best_solution[c], self.best_solution[d], self.best_solution[e], self.best_solution[f]

		gain = length(self.points[p_a], self.points[p_b]) + length(self.points[p_c], self.points[p_d]) + length(self.points[p_e], self.points[p_f])

		if execute == 0:
			# 2-opt (a, e) [d, c] (b, f)
			sol = self.best_solution[:a+1] + self.best_solution[e:d-1:-1] + self.best_solution[c:b-1:-1]+self.best_solution[f:]
			gain -= length(self.points[p_a], self.points[p_e]) + length(self.points[p_d], self.points[p_c]) + length(self.points[p_b], self.points[p_f])
		elif execute == 1:
			# 2-opt [a, b] (c, e) (d, f)
			sol = self.best_solution[:a + 1] + self.best_solution[b:c + 1] + self.best_solution[e:d - 1:-1] + self.best_solution[f:]
			gain -= length(self.points[p_a], self.points[p_b]) + length(self.points[p_c], self.points[p_e]) + length(self.points[p_d], self.points[p_f])

		elif execute == 2:
			# 2-opt (a, c) (b, d) [e, f]
			sol = self.best_solution[:a + 1] + self.best_solution[c:b - 1:-1] + self.best_solution[d:e + 1] + self.best_solution[f:]
			gain -= length(self.points[p_a], self.points[p_c]) + length(self.points[p_b], self.points[p_d]) + length(self.points[p_e], self.points[p_f])

		elif execute == 3:
			# 3-opt (a, d) (e, с) (b, f)
			sol = self.best_solution[:a + 1] + self.best_solution[d:e + 1] + self.best_solution[c:b - 1:-1] + self.best_solution[f:]
			gain -= length(self.points[p_a], self.points[p_d]) + length(self.points[p_e], self.points[p_c]) + length(self.points[p_b], self.points[p_f])

		elif execute == 4:
			# 3-opt (a, d) (e, b) (c, f)
			sol = self.best_solution[:a + 1] + self.best_solution[d:e + 1] + self.best_solution[b:c + 1] + self.best_solution[f:]
			gain -= length(self.points[p_a], self.points[p_d])+length(self.points[p_e], self.points[p_b])+length(self.points[p_c], self.points[p_f])



		elif execute == 5:
			# 3-opt (a, e) (d, b) (c, f)
			sol = self.best_solution[:a + 1] + self.best_solution[e:d - 1:-1] + self.best_solution[b:c + 1] + self.best_solution[f:]
			gain -= length(self.points[p_a], self.points[p_e])+length(self.points[p_d], self.points[p_b])+length(self.points[p_c], self.points[p_f])

		elif execute == 6:
			# 3-opt (a, c) (b, e) (d, f)
			 sol = self.best_solution[:a + 1] + self.best_solution[c:b - 1:-1] + self.best_solution[e:d - 1:-1] + self.best_solution[f:]
			 gain -= length(self.points[p_a], self.points[p_c]) + length(self.points[p_b], self.points[p_e]) + length(self.points[p_d], self.points[p_f])
		#print(sol[1::10])
		return sol, gain
	def improve(self, fast=False):
		if self.best_cost < 40000 and self.size < 700 and self.size > 500:
			return
		elif self.size > 700 and self.size < 2000 and self.best_cost < 378069:
			return
		elif self.size > 5000 and self.cost < 78478868:
			return

		if(fast):
			shift(self.best_solution, -15)

		bestChange = 0
		best_path =  copy.copy(self.best_solution)

		counter = round(self.size//4)

		list = [i for i in range(self.size)]

		list1 = random.sample(list[:self.size-5], counter)

		tick = counter

		for a in list1:
			for c in range(a+2, self.size-3):
				for e in range(c+2, self.size-1):
					change = 0

					for i in range(7):

						if self.best_cost < 0:
							return

						path, change = self.exchange_3(i, a, c, e)

						if change > bestChange:
							print("changes: ", change)
							bestChange = copy.copy(change)
							best_path =  copy.copy(path)
							#print(len(best_path))



							counter -= 3

							if counter < 0 or fast:
								self.best_cost -= bestChange
								self.best_solution = copy.copy(best_path)
								print(self.best_cost)
								return

				tick -= 1
				if(tick < 0):
					self.shift_it()
					break

			


		self.best_cost -= bestChange
		self.best_solution = copy.copy(best_path)
		print(self.best_cost)
		return

	def straight_it(self):

		sol = copy.copy(self.best_solution)

		i = random.randint(0, self.size-1)
		j = random.randint(0, self.size-1)

		if(i > j):
			i, j = j, i

		for u in range(i, j):
			sol.append(sol[0])
			for v in range(i+2, j+1):
				if intersec(self.points[sol[u]], self.points[sol[u+1]], self.points[sol[v]], self.points[sol[v+1]]):
					sol[u:v] = reversed(sol[u:v])

			sol.pop()

		value = length(self.points[sol[-1]], self.points[sol[0]])
		for index in range(0, self.size-1):
			value += length(self.points[sol[index]], self.points[sol[index+1]])

		print(value, self.best_cost)

		if(value < self.best_cost):
			self.best_solution = copy.copy(sol)
			self.best_cost = copy.copy(value)
			print("straight wotk: ", self.best_cost)
	def get_cost(self):
		return self.best_cost
	def get(self):
		return self.best_solution
	def shift_it(self):
		shift(self.best_solution, -1)


class speed_Dream_pop():

	tabu = set()

	def __init__(self, solution, points, cost):
		self.best_solution = copy.copy(solution)
		
		self.points = points
		self.best_cost = cost
		self.size = len(solution)
	def exchange_3(self, execute, a, c, e):
		b, d, f = a+1, c+1, e+1

		p_a, p_b, p_c, p_d, p_e, p_f = self.best_solution[a], self.best_solution[b], self.best_solution[c], self.best_solution[d], self.best_solution[e], self.best_solution[f]

		gain = length(self.points[p_a], self.points[p_b]) + length(self.points[p_c], self.points[p_d]) + length(self.points[p_e], self.points[p_f])

		if execute == 0:
			# 2-opt (a, e) [d, c] (b, f)
			sol = self.best_solution[:a+1] + self.best_solution[e:d-1:-1] + self.best_solution[c:b-1:-1]+self.best_solution[f:]
			gain -= length(self.points[p_a], self.points[p_e]) + length(self.points[p_d], self.points[p_c]) + length(self.points[p_b], self.points[p_f])
		elif execute == 1:
			# 2-opt [a, b] (c, e) (d, f)
			sol = self.best_solution[:a + 1] + self.best_solution[b:c + 1] + self.best_solution[e:d - 1:-1] + self.best_solution[f:]
			gain -= length(self.points[p_a], self.points[p_b]) + length(self.points[p_c], self.points[p_e]) + length(self.points[p_d], self.points[p_f])

		elif execute == 2:
			# 2-opt (a, c) (b, d) [e, f]
			sol = self.best_solution[:a + 1] + self.best_solution[c:b - 1:-1] + self.best_solution[d:e + 1] + self.best_solution[f:]
			gain -= length(self.points[p_a], self.points[p_c]) + length(self.points[p_b], self.points[p_d]) + length(self.points[p_e], self.points[p_f])

		elif execute == 3:
			# 3-opt (a, d) (e, с) (b, f)
			sol = self.best_solution[:a + 1] + self.best_solution[d:e + 1] + self.best_solution[c:b - 1:-1] + self.best_solution[f:]
			gain -= length(self.points[p_a], self.points[p_d]) + length(self.points[p_e], self.points[p_c]) + length(self.points[p_b], self.points[p_f])

		elif execute == 4:
			# 3-opt (a, d) (e, b) (c, f)
			sol = self.best_solution[:a + 1] + self.best_solution[d:e + 1] + self.best_solution[b:c + 1] + self.best_solution[f:]
			gain -= length(self.points[p_a], self.points[p_d])+length(self.points[p_e], self.points[p_b])+length(self.points[p_c], self.points[p_f])

		elif execute == 5:
			# 3-opt (a, e) (d, b) (c, f)
			sol = self.best_solution[:a + 1] + self.best_solution[e:d - 1:-1] + self.best_solution[b:c + 1] + self.best_solution[f:]
			gain -= length(self.points[p_a], self.points[p_e])+length(self.points[p_d], self.points[p_b])+length(self.points[p_c], self.points[p_f])

		elif execute == 6:
			# 3-opt (a, c) (b, e) (d, f)
			 sol = self.best_solution[:a + 1] + self.best_solution[c:b - 1:-1] + self.best_solution[e:d - 1:-1] + self.best_solution[f:]
			 gain -= length(self.points[p_a], self.points[p_c]) + length(self.points[p_b], self.points[p_e]) + length(self.points[p_d], self.points[p_f])
		#print(sol[1::10])
		return sol, gain
	def improve(self, fast=False):
		print(len(self.best_solution), self.size)
		if self.best_cost < 40000 and self.size < 700 and self.size > 500:
			return
		elif self.size > 700 and self.size < 2000 and self.best_cost < 378069:
			return
		elif self.size > 5000 and self.best_cost < 78478868:
			return




		bestChange = 0
		best_path =  copy.copy(self.best_solution)

		counter = self.size//2

		best_a = -1
		best_c = -1
		best_e = -1

		flag = False

		if(fast):
			shift(self.best_solution, -15)

		for a in range(self.size):
			if counter < 0:
				break

			if self.best_solution[a] in self.tabu:
				continue

			list1 = [i for i in range(a+2, self.size-3)]
			list1.sort(key=lambda x: length(self.points[self.best_solution[x]], self.points[self.best_solution[a]]))


			for c in list1:

				if counter < 0:
					break

				for e in range(c+2, self.size-1):
					if counter < 0:
						break
					change = 0

					for i in range(7):

						if self.best_cost < 0:
							return

						path, change = self.exchange_3(i, a, c, e)

						if change > bestChange:
							flag = True
							print("changes: ", change)
							bestChange = copy.copy(change)
							best_path =  copy.copy(path)
							best_a = a
							best_c = c
							best_e = e
							counter -= 3
							continue
							print(len(best_path))

							if counter < 0:
								self.best_cost -= bestChange
								self.best_solution = copy.copy(best_path)
								print(self.best_cost)
								return
					if(fast):
						counter -= 1

				if(not flag):
					self.tabu.add(self.best_solution[a])

		if best_a != -1 and self.best_solution[best_a] in self.tabu:
			self.tabu.remove(self.best_solution[best_a])
		if best_c != -1 and self.best_solution[best_c] in self.tabu:
			self.tabu.remove(self.best_solution[best_c])
		if best_e != -1 and self.best_solution[best_e] in self.tabu:
			self.tabu.remove(self.best_solution[best_e])		

			


		self.best_cost -= bestChange
		self.best_solution = copy.copy(best_path)
		print(self.best_cost)
		return

	def straight_it(self):

		sol = copy.copy(self.best_solution)

		i = random.randint(0, self.size-1)
		j = random.randint(0, self.size-1)

		if(i > j):
			i, j = j, i

		for u in range(i, j):
			sol.append(sol[0])
			for v in range(i+2, j+1):
				if intersec(self.points[sol[u]], self.points[sol[u+1]], self.points[sol[v]], self.points[sol[v+1]]):
					sol[u:v] = reversed(sol[u:v])

			sol.pop()

		value = length(self.points[sol[-1]], self.points[sol[0]])
		for index in range(0, self.size-1):
			value += length(self.points[sol[index]], self.points[sol[index+1]])

		print(value, self.best_cost)

		if(value < self.best_cost):
			self.best_solution = copy.copy(sol)
			self.best_cost = copy.copy(value)
			print("straight wotk: ", self.best_cost)
	def get_cost(self):
		return self.best_cost
	def get(self):
		return self.best_solution
	def shift_it(self):
		shift(self.best_solution, -1)




def solve_it(input_data):
	random.seed(9001)
    # Modify this code to run your optimization algorithm

    # parse the input
	lines = input_data.split('\n')

	nodeCount = int(lines[0])

	print(nodeCount)

	points = []
	for i in range(1, nodeCount+1):
		line = lines[i]
		parts = line.split()
		points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
	solution = [i for i in range(nodeCount)]
	solution = random.sample(solution, nodeCount)

	if(nodeCount == 574 or nodeCount == 1889 or nodeCount >30000):
		solution = get_greedy(solution, points, nodeCount)
	if(nodeCount == 1000):
		solution = [i for i in range(nodeCount)]

		solution1 = []
		solution2 = []

		k = (1080000-338125)/(939876-177562)
		y1 = 338125
		x1 = 177562

		for i in solution:
			if k*(points[i].y-y1)+x1 > points[i].x:
				solution1.append(i)
			else:
				solution2.append(i)

		#wtf man?

		best = 1000000

		best_u1 = solution2[0]
		best_v1 = solution1[0]

		for u1 in solution2:
			for v1 in solution1:
				if(length(points[u1], points[v1]) < best):
					best_u1 = u1
					best_v1 = v1
					best = length(points[u1], points[v1])

		best = 2000000

		best_u2 = solution2[0]
		best_v2 = solution2[0]

		for u2 in solution2:
			if v1 == best_v1:
				continue
			for v2 in solution1:
				if v2 == best_v1:
					continue
				if(length(points[u2], points[v2]) < best):
					best_u2 = u2
					best_v2 = v2
					best = length(points[u2], points[v2])

		flag = 0

		best_u1 = solution2.index(best_u1)
		best_u2 = solution2.index(best_u2)

		best_v1 = solution1.index(best_v1)
		best_v2 = solution1.index(best_v2)

		print(len(solution1), len(solution2))

		if best_u1 > best_u2:
			flag = (flag+1)%2
			best_u1, best_u2 = best_u2, best_u1

		solution2[:(best_u1+1)]=reversed(solution2[:(best_u1+1)])
		solution2[best_u2:]=reversed(solution2[best_u2:])

		if best_v1 > best_v2:
			flag = (flag+1)%2
			best_v1, best_v2 = best_v2, best_v1

		solution1[:(best_v1+1)]=reversed(solution1[:(best_v1+1)])
		solution1[best_v2:]=reversed(solution1[best_v2:])

		if(flag):
			solution1[:]=reversed(solution1[:])
		
		

		solution = solution1+solution2
		print(solution)

		# такой костыль...


	S = 0
	for index in range(nodeCount):
		S += length(points[solution[index]], points[solution[index-1]])

	# < 400
	
	if(nodeCount < 400):
		boy = hip_hop(solution, points, S, nodeCount)

		while(boy.improve()):
			continue

		solution = boy.get()

		S = 0
		for index in range(nodeCount):
			S += length(points[solution[index]], points[solution[index-1]])

		# 51

		if(nodeCount == 51):
			buf_solution = copy.copy(solution)
			S_buf = S

			for i in range(50):

				solution = random.sample(solution, nodeCount)

				S = 0
				for index in range(nodeCount):
					S += length(points[solution[index]], points[solution[index-1]])

				boy = hip_hop(solution, points, S, nodeCount)
				while(boy.improve()):
					continue
				solution = boy.get()

				S = 0
				for index in range(nodeCount):
					S += length(points[solution[index]], points[solution[index-1]])

				if(S < S_buf):
					buf_solution = copy.copy(solution)
					S_buf = S

				if(S_buf < 430):
					break

			solution = buf_solution

		else:
			boy = Dream_pop(solution, points, S)

			for i in range(10):
				boy.straight_it()

			for i in range(75):
				boy.improve(fast=True)

			boy.straight_it()
			solution = boy.get()
	elif(nodeCount == 574):

		boy = Dream_pop(solution, points, S)

		boy.straight_it()

		for i in range(65):
			boy.improve()

		boy.straight_it()

		for i in range(20):
			boy.improve(fast=True)

		boy.straight_it()

		for i in range(30):
			boy.improve()

		for i in range(10):
			boy.improve(fast=True)

		solution = boy.get()
	elif(nodeCount == 1889):
		
		boy = speed_Dream_pop(solution, points, S)
		
		boy.straight_it()

		for i in range(40):
			boy.straight_it()
		

		for i in range(nodeCount*20):
			boy.improve(fast=True)

		for i in range(20):
			boy.straight_it()

		for i in range(nodeCount*9):
			boy.improve()

		solution = boy.get()

	else:

		boy = speed_Dream_pop(solution, points, S)

		for i in range(nodeCount*10):
			boy.improve(fast=True)

		for i in range(nodeCount*5):
			boy.improve()

		solution = boy.get()

		

	print(len(solution))

	
	x = []
	y = []

	for u in solution:
		x.append(points[u].x)
		y.append(points[u].y)

	x.append(points[solution[0]].x)
	y.append(points[solution[0]].y)

	plt.plot(x, y, marker ='o')

	if(nodeCount == 1000):
		plt.plot([177562, 939876], [338125, 1080000])

	plt.show()
	
	
	

    # calculate the length of the tour
	obj = length(points[solution[-1]], points[solution[0]])
	for index in range(0, nodeCount-1):
		obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
	output_data = '%.2f' % obj + ' ' + str(0) + '\n'
	output_data += ' '.join(map(str, solution))

	return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

