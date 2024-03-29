# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)

import collections
import heapq
import itertools
import queue
import copy


def search(maze, searchMethod):
    return {
        "bfs": bfs(maze),
        "dfs": dfs(maze),
        "greedy": greedy(maze),
        "astar": astar(maze),
    }.get(searchMethod, [])
   

def bfs(maze):
    num_states_explored = 0
    start_position = maze.getStart()
    end_position = maze.getObjectives()
    visited, queue = [], collections.deque([start_position])
    parent_map = {start_position:start_position}
    not_found = True
    while queue and not_found:
        num_states_explored += 1
        vertex = queue.popleft()
        visited.append((vertex[0], vertex[1]))
        if (vertex[0], vertex[1]) == end_position[0]:
            not_found = False
            break
        for neighbour in maze.getNeighbors(vertex[0], vertex[1]):
            if neighbour not in queue and neighbour not in visited :
                queue.append(neighbour)
                parent_map[neighbour] = (vertex[0], vertex[1])

    parent_list = []
    current = end_position[0]
    while current != start_position:
        parent_list.append(current)
        current = parent_map[current]
    parent_list.append(current)
    parent_list.reverse()
    return parent_list, num_states_explored


def dfs(maze):
    num_states_explored = 0
    start_position = maze.getStart()
    end_position = maze.getObjectives()
    visited, queue = list(), collections.deque([start_position])
    parent_map = {start_position: start_position}
    not_found = True
    while queue and not_found:
        num_states_explored += 1
        vertex = queue.pop()
        visited.append((vertex[0], vertex[1]))
        if (vertex[0], vertex[1]) == end_position[0]:
            not_found = False
            break
        for neighbour in maze.getNeighbors(vertex[0], vertex[1]):
            if neighbour not in queue and neighbour not in visited :
                queue.append(neighbour)
                parent_map[neighbour] = (vertex[0], vertex[1])

    # backtrack to find the trace
    parent_list = list()
    current = end_position[0]
    while current != start_position:
        parent_list.append(current)
        current = parent_map[current]
    parent_list.append(current)
    parent_list.reverse()
    return parent_list, num_states_explored


def greedy(maze):
    # initialization
    num_states_explored = 0
    start_position = maze.getStart()
    end_position = maze.getObjectives()
    # book keeping, which node is visited and what is in the priority queue
    visited, queue = [], [(abs(end_position[0][0]-start_position[0])+abs(end_position[0][1]-start_position[1]),(start_position))]
    parent_map = {start_position: start_position}
    not_found = True
    while queue and not_found:
        num_states_explored += 1
        vertex = heapq.heappop(queue)[1]
        visited.append((vertex[0], vertex[1]))
        if (vertex[0], vertex[1]) == end_position[0]:
            not_found = False
            break
        for neighbour in maze.getNeighbors(vertex[0], vertex[1]):
            if neighbour not in visited and neighbour not in (x[1] for x in queue):
                heapq.heappush(queue, (
                    abs(end_position[0][0] - neighbour[0]) + abs(end_position[0][1] - neighbour[1]), (neighbour[0], neighbour[1])))
                parent_map[neighbour] = (vertex[0], vertex[1])

    # backtrack to find the trace
    parent_list = list()
    current = end_position[0]
    while current != start_position:
        parent_list.append(current)
        current = parent_map[current]
    parent_list.append(current)
    parent_list.reverse()
    return parent_list, num_states_explored


def astar(maze):
    global globalpath
    global globallen
    global not_found
    # initialization
    num_states_explored = 0
    start_position = maze.getStart()
    end_position = maze.getObjectives()

    if (len(end_position) > 1):

        globalpath = []
        globallen = 0
        not_found =True
        visited = []
        num_states_explored = 0
        path = []
        astarMultiple(maze, start_position, end_position, visited, num_states_explored, path)

        return globalpath, globallen

    # book keeping, which node is visited and what is in the priority queue
    visited, queue = [], [(distance(start_position, end_position[0]), start_position, 0)]
    parent_map = {start_position: start_position}
    not_found = True
    while queue and not_found:
        num_states_explored += 1
        node = heapq.heappop(queue)
        vertex = node[1]
        visited.append(vertex)
        if (vertex[0], vertex[1]) == end_position[0]:
            not_found = False
            break
        for neighbour in maze.getNeighbors(vertex[0], vertex[1]):
            # if neighbour not in visited and neighbour not in (x[1] for x in queue):
            if neighbour not in visited:
                position_queue = [x[1] for x in queue]
                if neighbour not in position_queue:
                    heapq.heappush(queue, (distance(neighbour, end_position[0])+node[2]+1, neighbour, node[2]+1))
                    parent_map[neighbour] = (vertex[0], vertex[1])
                else:
                    index = position_queue.index(neighbour)
                    if (distance(neighbour, end_position[0])+node[2]+1) < queue[index][0]:
                        heapq.heappush(queue,
                                       (distance(neighbour, end_position[0]) + node[2] + 1, neighbour, node[2] + 1))
                        parent_map[neighbour] = (vertex[0], vertex[1])

    # backtrack to find the trace
    parent_list = list()
    current = end_position[0]
    while current != start_position:
        parent_list.append(current)
        current = parent_map[current]
    parent_list.append(current)
    parent_list.reverse()
    return parent_list, num_states_explored

def astarMultiple(maze, start_position, end_position, visited, num_states_explored, oldpath):

    # book keeping, which node is visited and what is in the priority queue
    queue = [(0, start_position, [start_position])]
    global not_found
    while queue:

        node = heapq.heappop(queue)
        path = node[2]
        (row, col) = (node[1][0], node[1][1])

        if node[1] not in visited:
            visited.append(node[1])
            if (row, col) in end_position:
                if len(end_position) == 1:
                    global globalpath
                    global globallen

                    if len(oldpath + path) < len(globalpath) or globalpath == []:
                        globalpath = oldpath + path
                        globallen = len(visited) + num_states_explored
                        # print("newlength!!!")
                        # print(len(globalpath))
                        not_found = False
                        return

                else:
                    new_visited = []
                    new_start_position = node[1]
                    new_end_position = copy.copy(end_position)
                    new_end_position.remove((row, col))

                    astarMultiple(maze, new_start_position, new_end_position, new_visited, len(visited) + num_states_explored,
                                      oldpath + path)
            if not not_found:
                return

            neighbors = maze.getNeighbors(row, col)
            for i, neighbor in enumerate(neighbors):
                position_queue = [x[1] for x in queue]
                if neighbor not in position_queue:
                    heapq.heappush(queue, (distance_sum(neighbor, end_position)+len(path)+1, neighbor, path + [neighbor]))
                else:
                    index = position_queue.index(neighbor)
                    if (distance_sum(neighbor, end_position)+len(path)+1) < queue[index][0]:
                        heapq.heappush(queue, (
                            distance_sum(neighbor, end_position) + len(path) + 1, neighbor, path + [neighbor]))

    return


def distance(start_pos, end_pos):
    dis = abs(end_pos[0] - start_pos[0]) + abs(end_pos[1] - start_pos[1])
    return dis
def distance_sum(current, end_pos):
    end_list = copy.copy(end_pos)
    min = -1
    minindex = 0
    total_dis=0
    while end_list:
        for i, item in enumerate(end_list):
            currentdis = distance(current, item)
            if currentdis < min or min == -1:
                minindex=i
                min = currentdis
        total_dis +=min
        min =-1
        current = end_list[minindex]
        del end_list[minindex]

    return  total_dis


def distance_max(current, end_pos):

    max = -1
    maxindex=0
    min = -1
    minindex = 0
    for i, item in enumerate(end_pos):
        currentdis = distance(current, item)
        if currentdis > max or max == -1:
            maxindex=i
            max = currentdis
        if currentdis < min or max == -1:
            minindex=i
            min = currentdis


    return  min+distance(end_pos[minindex],end_pos[maxindex])

