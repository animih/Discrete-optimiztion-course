#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

#capacity = 0

def find_opt_sum(items, number, capacity):

	Matrix_1 = [0 for i in range(capacity+1)]
	Matrix_2 = [0 for i in range(capacity+1)]

	for i in range(number):
		if(items[i].weight > capacity):
			continue
		for j in range(items[i].weight):
			Matrix_2[j] = Matrix_1[j]
		for j in range(items[i].weight, capacity+1):
			Matrix_2[j] = max(Matrix_1[j], Matrix_1[j - items[i].weight] + items[i].value)

		for j in range(capacity+1):
			Matrix_1[j] = Matrix_2[j]


	return Matrix_2[capacity]


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

	opt_sum = find_opt_sum(items, item_count, capacity)
	value = opt_sum

	#print(opt_sum)

	all = len(items)

	i = 0
	while(i != item_count):
		item = items[item_count-1-i]
		items.pop(item_count-1-i)
		item_count -= 1
		#print(find_opt_sum(items, item_count, capacity), i)
		if(opt_sum != find_opt_sum(items, item_count, capacity)):
			items.append(item)
			taken[item_count-i] = 1
			item_count += 1
			i += 1

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

