#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import random
import copy
import matplotlib.pyplot as plt
from collections import namedtuple

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def makepair(i, j):
    if(i > j):
        return (j, i)
    else:
        return (i, j)

class ECS:
    best_route = []

    iteration = 0

    alpha = 1
    betta = 3
    yamma = 5
    rho = 0.1
    q_0 = 0.75

    T = 1200 

    pheromones = {}

    def __init__(self, customers, vehicle_count, vehicle_capacity):
        self.vehicle_count = vehicle_count
        self.vehicle_capacity = vehicle_capacity
        self.customers = customers

        self.build_trivial()

        self.pheromones[(0, 0)] = 0;

    def build_trivial(self):

        remaining_customers = set(self.customers)
        remaining_customers.remove(self.customers[0])

        for v in range(0, self.vehicle_count):
            # print "Start Vehicle: ",v
            self.best_route.append(self.customers[0].index)
            capacity_remaining = self.vehicle_capacity
            while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
                used = set()
                order = sorted(remaining_customers, key=lambda customer: -customer.demand*len(self.customers) + customer.index)
                for customer in order:
                    if capacity_remaining >= customer.demand:
                        capacity_remaining -= customer.demand
                        # print '   add', ci, capacity_remaining
                        used.add(customer)
                        self.best_route.append(customer.index)
                remaining_customers -= used

        self.best_value = 0

        print(len(self.best_route))
        print(self.best_route)
        #print("there")
        #assert len(self.best_route) == len(self.customers)-1 +  self.vehicle_count

        for v in range(len(self.best_route)):
            self.best_value += length(self.customers[self.best_route[v-1]], self.customers[self.best_route[v]])

        for v in range(len(self.customers)):
            for u in range(len(self.customers)):
                self.pheromones[makepair(v, u)] = 0.04*1/self.best_value

        for v in range(len(self.best_route)):
            if(self.best_route[v-1] != (self.best_route[v])):
                self.pheromones[makepair(self.best_route[v], self.best_route[v-1])] = 1/self.best_value


        self.best_counter = self.vehicle_count*3
    def get_pheromone(self, i, j):
        if makepair(i, j) in self.pheromones:
            return self.pheromones[makepair(i, j)]
        else:
            return 0

    def get_param(self, i, j):
        if(i == j):
            return 0

        if(length(self.customers[i], self.customers[j]) == 0):
            return 1

        return self.get_pheromone(i, j)**self.alpha * \
        1/length(self.customers[i], self.customers[j])**self.betta * \
        (length(self.customers[i], self.customers[0])+length(self.customers[j], self.customers[0])-\
                        2*length(self.customers[i], self.customers[j]) + 2* \
                        abs(length(self.customers[i], self.customers[0])-length(self.customers[j], self.customers[0])))**self.yamma
    # пишу мега-универслаьно 2-3 опт, но на деле пока что исопльзую только 2 опт
    def exchange_3(self, route, execute, a, c, e):
        b, d, f = a+1, c+1, e+1

        p_a, p_b, p_c, p_d, p_e, p_f = route[a], route[b], route[c], route[d], route[e], route[f]

        gain = length(self.customers[p_a], self.customers[p_b]) + length(self.customers[p_c], self.customers[p_d]) + length(self.customers[p_e], self.customers[p_f])

        if execute == 0:
            # 2-opt (a, e) [d, c] (b, f)
            sol = route[:a+1] + route[e:d-1:-1] + route[c:b-1:-1]+route[f:]
            gain -= length(self.customers[p_a], self.customers[p_e]) + length(self.customers[p_d], self.customers[p_c]) + length(self.customers[p_b], self.customers[p_f])
        elif execute == 1:
            # 2-opt [a, b] (c, e) (d, f)
            sol = route[:a + 1] + route[b:c + 1] + route[e:d - 1:-1] + route[f:]
            gain -= length(self.customers[p_a], self.customers[p_b]) + length(self.customers[p_c], self.customers[p_e]) + length(self.customers[p_d], self.customers[p_f])

        elif execute == 2:
            # 2-opt (a, c) (b, d) [e, f]
            sol = route[:a + 1] + route[c:b - 1:-1] + route[d:e + 1] + route[f:]
            gain -= length(self.customers[p_a], self.customers[p_c]) + length(self.customers[p_b], self.customers[p_d]) + length(self.customers[p_e], self.customers[p_f])

        elif execute == 3:
            # 3-opt (a, d) (e, с) (b, f)
            sol = route[:a + 1] + route[d:e + 1] + route[c:b - 1:-1] + route[f:]
            gain -= length(self.customers[p_a], self.customers[p_d]) + length(self.customers[p_e], self.customers[p_c]) + length(self.customers[p_b], self.customers[p_f])

        elif execute == 4:
            # 3-opt (a, d) (e, b) (c, f)
            sol = route[:a + 1] + route[d:e + 1] + route[b:c + 1] + route[f:]
            gain -= length(self.customers[p_a], self.customers[p_d])+length(self.customers[p_e], self.customers[p_b])+length(self.customers[p_c], self.customers[p_f])

        elif execute == 5:
            # 3-opt (a, e) (d, b) (c, f)
            sol = route[:a + 1] + route[e:d - 1:-1] + route[b:c + 1] + route[f:]
            gain -= length(self.customers[p_a], self.customers[p_e])+length(self.customers[p_d], self.customers[p_b])+length(self.customers[p_c], self.customers[p_f])

        elif execute == 6:
            # 3-opt (a, c) (b, e) (d, f)
             sol = route[:a + 1] + route[c:b - 1:-1] + route[e:d - 1:-1] + route[f:]
             gain -= length(self.customers[p_a], self.customers[p_c]) + length(self.customers[p_b], self.customers[p_e]) + length(self.customers[p_d], self.customers[p_f])
        #print(sol[1::10])
        return sol, gain

    def exchange_2(self, route, a, c, end):
        b, d = a+1, c+1

        

        p_a, p_b, p_c, p_d = route[a], route[b], route[c], route[d]

        if(c == end-1):
            p_d = p_a

        sol = copy.copy(route)

        sol[b:d] = reversed(sol[b:d])

        gain = length(self.customers[p_a], self.customers[p_b]) + length(self.customers[p_c], self.customers[p_d])

        gain -= length(self.customers[p_a], self.customers[p_c]) + length(self.customers[p_b], self.customers[p_d])

        return sol, gain


    def run_ant(self):

        

        solution = []

            #print("new ant")
        remaining_customers = set(self.customers)
        remaining_customers.remove(self.customers[0])

        solution = [0]
        i = 0;

        pos = 0
        capacity_remaining = self.vehicle_capacity
        counter = 1
        while(remaining_customers):

            while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
                order = sorted(remaining_customers, key=lambda customer: -self.get_param(solution[i], customer.index))


                out = 0
                end = len(order)
                while(out != end):
                    if order[out].demand > capacity_remaining:
                        order.pop(out)
                        end -= 1
                    else:
                        out += 1

                q = random.random()



               	if(len(order) == 1):
                    i += 1
                    remaining_customers.remove(self.customers[order[0].index])
                    capacity_remaining = self.vehicle_capacity
                    solution.append(order[0].index)
                    break

                if(q < self.q_0):
                    remaining_customers.remove(self.customers[order[0].index])
                    capacity_remaining -= order[0].demand
                    i += 1
                    solution.append(order[0].index)
                else:
                    norm = 0

                    for cus in order:
                        norm += self.get_param(solution[i], cus.index)

                    if norm == 0:
                        print(solution[i], order)

                    assert norm != 0

                    r = random.random()

                    s = 0;
                    j = -1

                    while(r > s/norm):
                        j += 1
                        s += self.get_param(solution[i], order[j].index)


                        
                    remaining_customers.remove(order[j])
                    capacity_remaining -= order[j].demand
                    i += 1
                    solution.append(order[j].index)

                if(solution[i] == 0):
                   break

            solution.append(0)
            i += 1

            #print(solution, i)
            #return
                
            if(solution[i] == 0):
                #remaining_customers.add(self.customers[0])
                counter += 1
                capacity_remaining = self.vehicle_capacity
                if (i-pos) < 4:
                    pos = i
                    continue

                for l in range((i-pos)):

                    best_change = 0
                    best_path = copy.copy(solution)

                    for a in range(pos, i-2):
                        for c in range(a+2, i):
                            change = 0
                            path, change = self.exchange_2(solution, a, c, i)

                            if change > best_change:
                                best_path = copy.copy(path)
                                best_change = change

                        solution = copy.copy(best_path)

                pos = i

        value = 0

        for v in range(len(solution)):
            value += length(self.customers[solution[v-1]], self.customers[solution[v]])
        
        if(counter > self.best_counter):
        	self.update_pheromones()
        	self.T *= 0.95
        	return
		
        self.best_counter = counter


        if(value < self.best_value):

            print(counter)
            self.best_value = value
            self.best_route = copy.copy(solution) 

            for v in range(len(solution)):
                self.pheromones[makepair(solution[v-1], solution[v])] += self.rho/value;

        elif(random.random() < math.exp((self.best_value - value)/self.T)):

            for v in range(len(solution)):
                self.pheromones[makepair(solution[v-1], solution[v])] += self.rho/value;
                

        self.update_pheromones()
        self.T *= 0.95
        


    def update_pheromones(self):

        for key in self.pheromones.keys():
            self.pheromones[key] *= (1-self.rho)


        if(self.iteration == 0):
            for v in range(len(self.best_route)):
                self.pheromones[makepair(self.best_route[v-1], self.best_route[v])] += self.rho/self.best_value

        self.iteration = (self.iteration+1)%5



    def get_route(self):
        return self.best_route



def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

    #the depot is always the first customer in the input
    depot = customers[0] 


    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = []

    print(vehicle_count)

    solver = ECS(customers, vehicle_count, vehicle_capacity)

    for i in range(round(math.sqrt(len(customers))*12)):
        solver.run_ant()

        current_sollution = solver.get_route()
        '''
        x = []
        y = []

        for i in current_sollution:
            x.append(customers[i].x)
            y.append(customers[i].y)

        plt.draw()
        plt.plot(x, y, marker = 'o')
        plt.pause(0.001)
        plt.clf()
    	'''
    '''
    x = []
    y = []

    for i in current_sollution:
        x.append(customers[i].x)
        y.append(customers[i].y)

    plt.plot(x, y, marker = 'o')
    plt.show()
	'''
    print(current_sollution)

    
    remaining_customers = set(customers)
    remaining_customers.remove(depot)

    start = 0
    end = 0
    
    for v in range(0, vehicle_count):

    	if(end == len(current_sollution)):
    		vehicle_tours.append([])
    		continue

    	start = end
    	end += 1

    	if(end == len(current_sollution)):
    		vehicle_tours.append([])
    		continue

    	while(current_sollution[end] != 0):
    		end += 1

    	vehicle_tours.append(current_sollution[start+1:end])
    	#vehicle_tours[v].append(0)

    # calculate the cost of the solution; for each vehicle the length of the route
    obj = 0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 0:
            obj += length(depot,customers[vehicle_tour[0]])
            for i in range(0, len(vehicle_tour)-1):
                obj += length(customers[vehicle_tour[i]],customers[vehicle_tour[i+1]])
            obj += length(customers[vehicle_tour[-1]],depot)

    print(vehicle_tours)

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        if(len(vehicle_tours[v]) == 0):
            outputData += '0 0' + '\n'
            continue
        outputData += str(depot.index) + ' ' + ' '.join([str(cus) for cus in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'
    return outputData


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

