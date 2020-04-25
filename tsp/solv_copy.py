#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import copy
import random
import matplotlib.pyplot as plt

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



class hip_hop:

	def __init__(self, solution, points, size):
		self.solution = solution
		self.points = points
		self.size = size

		self.cost = length(points[solution[-1]], points[solution[0]])
		for index in range(0, self.size-1):
			self.cost += length(points[solution[index]], points[solution[index+1]])
	def fix_it(self):

		tmp_sol = copy.copy(self.solution)
		tabu = set()

		counter = 0

		pa = random.randint(0, (self.size-1))

		while counter < math.sqrt(self.size) and len(tabu) < self.size-1:

			if pa == self.size-1:
				pb = 0
				min_len = length(self.points[tmp_sol[pa]], self.points[tmp_sol[pb]])
			else:
				pb = pa + 1
				min_len = length(self.points[tmp_sol[pa]], self.points[tmp_sol[pb]])

			pd = pa

			for i in range(self.size):
				if i == pa or i == pb or i == (pb+1) or i in tabu:
					continue
				elif(min_len > length(self.points[tmp_sol[pb]], self.points[tmp_sol[i]])):
					pd = i
					min_len = length(self.points[tmp_sol[pb]], self.points[tmp_sol[i]])

			if pa == pd:
				print("Ohhhhhhhh")
				break

			tabu.add(tmp_sol[pd])

			shift(tmp_sol, -1*pa)

			# fixing shit after shifting shit in fixing shit method in a shitty class. By the way, shhhhhiiitttt!

			if pd < pa:
				pd = self.size + pd - pa
			else:
				pd -= pa

			pc = pd - 1
			pa = 0
			pb = 1

			tmp_sol[pa:pc] = reversed(tmp_sol[pa:pc])

			value = length(self.points[tmp_sol[-1]], self.points[tmp_sol[0]])
			for index in range(0, self.size-1):
				value += length(self.points[tmp_sol[index]], self.points[tmp_sol[index+1]])
			if(value < self.cost):
				self.cost = copy.copy(value)
				self.solution = copy.copy(tmp_sol)
				print("accepted : ", counter)
				print("value = ", self.cost)
				counter += 6
			else:
				#print("denied : ", counter)
				counter += 1

	def straight_it(self):

		sol = copy.copy(self.solution)

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

		if(value < self.cost):
			self.solution = copy.copy(sol)
			self.cost = copy.copy(value)



	def get_solution(self):
		return self.solution

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

		list.remove(min_u)
		current = min_u

	return solution







def solve_it(input_data):
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

	coef = math.log(nodeCount)**2/500

	it = 0
	T = CoolThisShit(it, coef)

	E_buf = length(points[solution[-1]], points[solution[0]])
	for index in range(0, nodeCount-1):
		E_buf += length(points[solution[index]], points[solution[index+1]]) 

	if(nodeCount > 500):
		coef *= 2
		solution = get_greedy(solution, points, nodeCount)
		T = 0.05

	if(nodeCount > 5000):
		coef *= 20


	while(T > 0.05):

		buf_solution = copy.copy(solution)
		value = random.random()
 
		solution = GetCandidate(solution, nodeCount)

		E = length(points[solution[-1]], points[solution[0]])

		for index in range(0, nodeCount-1):
			E += length(points[solution[index]], points[solution[index+1]])
		if(E < E_buf):
			E_buf = E
		elif(value < math.exp((E_buf-E)/T)):
			E_buf = E
		else:
			solution = copy.copy(buf_solution)


		it += 1
		T = CoolThisShit(it, coef)
		print(T)

	
	print(solution, E_buf)

	x = []
	y = []

	for u in solution:
		x.append(points[u].x)
		y.append(points[u].y)

	x.append(points[solution[0]].x)
	y.append(points[solution[0]].y)

	plt.plot(x, y, marker ='o')
	plt.show()

	Boy = hip_hop(solution, points, nodeCount)
	for i in range(int(55*math.sqrt(nodeCount))):
		Boy.fix_it()
		if(i % 5 == 0):
			Boy.straight_it()

	solution = Boy.get_solution()

	E_buf = length(points[solution[-1]], points[solution[0]])
	for index in range(0, nodeCount-1):
		E_buf += length(points[solution[index]], points[solution[index+1]]) 
	print(solution, E_buf)

	x = []
	y = []

	for u in solution:
		x.append(points[u].x)
		y.append(points[u].y)

	x.append(points[solution[0]].x)
	y.append(points[solution[0]].y)

	plt.plot(x, y, marker ='o')
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

