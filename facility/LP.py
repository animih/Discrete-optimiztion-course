#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import random
import pulp as p 

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
	return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def estimate(solution, facilites, customers):

	s = 0

	is_open = [0 for i in range(len(facilites))]

	for i in range(len(solution)):
		s += length(facilites[solution[i]].location, customers[i].location)
		is_open[solution[i]] = 1

	for i in range(len(is_open)):
		s += facilites[i].cost*is_open[i]

	return s


def solve_it(input_data):
	# Modify this code to run your optimization algorithm

	# parse the input
	lines = input_data.split('\n')

	parts = lines[0].split()
	facility_count = int(parts[0])
	customer_count = int(parts[1])
	
	facilities = []
	for i in range(1, facility_count+1):
		parts = lines[i].split()
		facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

	customers = []
	for i in range(facility_count+1, facility_count+1+customer_count):
		parts = lines[i].split()
		customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

	# first. let me with liner relaxation

	prob = p.LpProblem("MinValue", p.LpMinimize)

	y = [0] * facility_count
	cost = [facilities[i].setup_cost for i in range(facility_count)]
	x = [0] * (facility_count*customer_count)
	dist = []
	demands = []

	for u in range(facility_count):
		for v in range(customer_count):
			dist.append(length(facilities[u].location, customers[v].location))
			demands.append(customers[v].demand)


	for i in range(len(x)):
		x[i] = p.LpVariable("x_{}".format(i), lowBound = 0, upBound = 1)
	for i in range(len(y)):
		y[i] = p.LpVariable("y_{}".format(i), lowBound = 0, upBound = 1)

	# objective function
	prob += p.lpDot(dist, x) + p.lpDot(cost, y)

	#constraints
	for i in range(facility_count):
		prob += facilities[i].capacity*y[i] - \
		p.lpDot(demands[i*customer_count : (i*customer_count+customer_count)], \
		x[i*customer_count : (i*customer_count+customer_count)]) >= 0

	# client has to be satisfied
	for i in range(customer_count):
		prob += p.lpSum(x[i::customer_count]) >= 1

	status = prob.solve()

	res = {}

	for i in range(customer_count):
		res[i] = []

	for u in range(facility_count):
		for v in range(customer_count):
			if p.value(x[u*customer_count+v]) > 0:
				res[v].append(u)
				res[v].append(p.value(x[u*customer_count+v]))

	open_wr = []

	for u in range(facility_count):
		open_wr.append(p.value(y[u]))

	print("linear relaxation gives:")

	print(res)
	print(open_wr)


	# now it's time for branch and bound, kids

	to_explore = {}
	best_so_far = 9999999999999999

	for u in res.keys():
		if(len(res[u]) > 2):
			to_explore[u] = res[u][::2]

	print(to_explore)


	# build a trivial solution
	# pack the facilities one by one until all the customers are served
	solution = [-1]*len(customers)
	capacity_remaining = [f.capacity for f in facilities]

	facility_index = 0
	for customer in customers:
		if capacity_remaining[facility_index] >= customer.demand:
			solution[customer.index] = facility_index
			capacity_remaining[facility_index] -= customer.demand
		else:
			facility_index += 1
			assert capacity_remaining[facility_index] >= customer.demand
			solution[customer.index] = facility_index
			capacity_remaining[facility_index] -= customer.demand

	used = [0]*len(facilities)
	for facility_index in solution:
		used[facility_index] = 1

	# calculate the cost of the solution
	obj = sum([f.setup_cost*used[f.index] for f in facilities])
	for customer in customers:
		obj += length(customer.location, facilities[solution[customer.index]].location)

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
		print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

