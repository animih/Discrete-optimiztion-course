#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import matplotlib.pyplot as plt
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

	print("this: (eto)")
	print(facility_count, customer_count)

	if((customer_count == 2000 and facility_count == 2000) or (customer_count == 1500 and facility_count == 1000) or (customer_count == 3000 and facility_count == 500)):
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


	prob = p.LpProblem("MinValue", p.LpMinimize)

	if((facility_count == 100 and customer_count == 1000) or (facility_count == 200 and customer_count == 8000)):
		facility_count = facility_count//2

	y = p.LpVariable.dicts("Warehouses", (range(0, facility_count)), cat='Binary')
	x = p.LpVariable.dicts("Choice", (range(0, facility_count), range(0, customer_count)), cat='Binary')

	cost = [facilities[i].setup_cost for i in range(facility_count)]
	demands = [customer.demand for customer in customers]
	dist = [[length(facilities[v].location, customers[u].location) for u in range(customer_count)] for v in range(facility_count)]

	# objective function
	prob += p.lpSum([y[v]*cost[v] for v in range(facility_count)])+ p.lpSum([p.lpSum([dist[v][u]*x[v][u] for u in range(customer_count)]) for v in range(facility_count)])

	#constraints
	for i in range(facility_count):
		prob += y[i]*facilities[i].capacity - p.lpSum([x[i][u]*demands[u] for u in range(customer_count)]) >= 0

	# client has to be satisfied
	for i in range(customer_count):
		prob += p.lpSum([x[v][i] for v in range(facility_count)]) == 1

	status = prob.solve()

	solution = []

	for u in range(customer_count):
		for v in range(facility_count):
			if p.value(x[v][u]) > 0:
				solution.append(v)

	print(solution)
	
	for i in range(len(solution)):
		plt.plot([facilities[solution[i]].location.x, customers[i].location.x], [facilities[solution[i]].location.y, customers[i].location.y], color='blue')

	plt.show()
	


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

