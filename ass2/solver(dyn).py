#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])
Node = namedtuple("Node", ['value', 'room', 'estimate'])


def oh_my(u):
	i = 1
	cmp = 2

	while cmp <= u:
		i += 1
		cmp *= 2

	return i


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
	lines = input_data.split('\n')

	firstLine = lines[0].split()
	item_count = int(firstLine[0])
	capacity = int(firstLine[1])

	items = []
	

	for i in range(1, item_count+1):
		line = lines[i]
		parts = line.split()
		items.append(Item(i-1, int(parts[0]), int(parts[1])))
		

    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
	value = 0
	weight = 0
	taken = [0]*len(items)

	if(item_count > 250):
		output_data = str(value) + ' ' + str(0) + '\n'
		output_data += ' '.join(map(str, taken))
		return output_data

	Shit_matrix = []

	Shit_matrix.append([])

	for j in range(capacity+1):
		Shit_matrix[0].append(0)





	for i in range(item_count):
		Shit_matrix.append([])
		if (items[i].weight > capacity):
			for j in range(capacity+1):
				Shit_matrix[i+1].append(Shit_matrix[i][j])
			continue
		
		for j in range(items[i].weight):
			Shit_matrix[i+1].append(Shit_matrix[i][j])
		for j in range(items[i].weight, capacity+1):
			Shit_matrix[i+1].append(max(Shit_matrix[i][j], Shit_matrix[i][j-items[i].weight]+items[i].value))
	
	
	value = Shit_matrix[item_count][capacity]
	
	
	while(capacity != 0 and item_count !=0):
		if(Shit_matrix[item_count][capacity] != Shit_matrix[item_count-1][capacity]):
			capacity -= items[item_count-1].weight
			taken[item_count-1] = 1

		item_count -= 1
	


	#print(Shit_matrix)
	'''

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    '''

	
    
    # prepare the solution in the specified output format
	output_data = str(value) + ' ' + str(0) + '\n'
	output_data += ' '.join(map(str, taken))
	return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

