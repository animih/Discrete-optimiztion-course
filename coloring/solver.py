#!/usr/bin/python
# -*- coding: utf-8 -*-

import copy
import random

def ev_func(solution, edges, len, color):
    C = [0 for i in range(color)]
    E = [0 for i in range(color)]



    for i in range(len):

        C[solution[i]] += 1
        a = False
        for u in edges[i]:

            if(solution[i] == solution[u] and u > i):
                E[solution[i]] += 1
                a = True

        if(a):
            E[solution[i]] += 1

    s = 0

    for i in range(color):
        s += -C[i]**2 + 2*C[i]*E[i]

    return s



def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    vertex = [i for i in range(node_count)]
    solution = [0 for i in range(node_count)]

    graph = {i:[] for i in range(node_count)}


    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        graph[int(parts[0])].append(int(parts[1]))
        graph[int(parts[1])].append(int(parts[0]))
        edges.append((int(parts[0]), int(parts[1])))

    # build a trivial solution
    # every node has its own color

    vertex.sort(key=lambda val: -len(graph[val]))

    min_color = 0
    while vertex:
        reduction = set()
        i = 0
        while(i < len(vertex)):

            if(vertex[i] in reduction):
                i += 1
                continue

            solution[vertex[i]] = min_color
            for j in graph[vertex[i]]:
                reduction.add(j)

            vertex.pop(i)

        min_color += 1

    
    if(node_count >= 500):
        output_data = str(node_count) + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, solution))

        return output_data
    

	# here comes local search woth penalty function

    min_value = ev_func(solution, graph, len(solution), min_color)

    print(solution, min_color, min_value)

    super_buf_solution = copy.copy(solution)

    while(True):

        buf_solution = copy.copy(solution)
        buf_value = min_value

        min_color -= 1

        for i in range(node_count):
            if(solution[i] == min_color):
                solution[i] -= 1

        # searching for local min (nemo)

        # euristic function

        C = [0 for i in range(min_color)]
        E = [0 for i in range(min_color)]

        for i in range(node_count):

            C[solution[i]] += 1
            flag = False

            for u in graph[i]:

                if(u > i and solution[i] == solution[u]):
                    E[solution[i]] += 1
                    flag = True

            if(flag):
                E[solution[i]] += 1


        min_value = 0
        for i in range(min_color):
            min_value += -C[i]**2 + 2*C[i]*E[i]

        move = True

        while(move):
            move = False

            for i in range(node_count):
                buf_color = solution[i]
                BUF_E_I = copy.copy(E[buf_color])

                for j in range(min_color):

                    BUF_E_J = copy.copy(E[j])

                    value = min_value + C[buf_color]**2 + C[j]**2 - 2 * E[buf_color] * C[buf_color] - 2 * E[j] * C[j]

                    C[buf_color] -= 1

                    solution[i] = j
                    C[j] += 1

                    Itse_in = False

                    for u in graph[i]:
                        Neih_not_in = True
                        if(solution[u] == buf_color):
                            Itse_in = True

                            for v in graph[u]:
                                if(v != i and solution[v] == buf_color):
                                    Neih_not_in = False
                                    break

                            if(Neih_not_in):
                                E[buf_color] -= 1

                    if(Itse_in):
                        E[buf_color] -= 1

                    Itse_in = False

                    for u in graph[i]:
                        Neih_not_in = True
                        if(solution[u] == solution[i]):
                            Itse_in = True

                            for v in graph[u]:
                                if(v != i and solution[v] == solution[i]):
                                    Neih_not_in = False
                                    break

                            if(Neih_not_in):
                                E[solution[i]] += 1

                    if(Itse_in):
                        E[solution[i]] += 1


                    value += -C[j]**2 - C[buf_color]**2 + 2*E[j]*C[j] + 2* E[buf_color] * C[buf_color]

                    if(value < min_value):
                        min_value = value
                        buf_color = solution[i]
                        BUF_E_I = copy.copy(E[buf_color])
                        BUF_E_J = copy.copy(E[j])
                        move = True
                    else:
                        C[solution[i]] -= 1
                        solution[i] = buf_color
                        E[buf_color] = BUF_E_I
                        E[j] = BUF_E_J
                        C[buf_color] += 1

        if(min_value >= buf_value):
            solution = copy.copy(buf_solution)
            min_color += 1
            break
        else:
            print(min_value)



    flag = 0
    for i in range(node_count):
        for u in graph[i]:
            if u < i:
                continue
            if(solution[i] == solution[u]):
                solution = super_buf_solution
                print("violation!")
                flag =1
                break;
        if(flag):
            break;

    print(solution, min_color, min_value)

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

