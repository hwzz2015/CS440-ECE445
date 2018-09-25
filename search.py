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


    # initialization
    num_states_explored = 0
    start_position = maze.getStart()
    end_position = maze.getObjectives()

    if (len(end_position) > 1):
        return astarMultiple(maze)
    elif (len(end_position) == 1):
        return astarsingle(maze,start_position,end_position[0])





def astarMultiple(maze):

    # initialization
    num_states_explored = 0
    start_position = maze.getStart()
    end_position = maze.getObjectives()

    mst_dis = findMST(maze,end_position)

    # book keeping, which node is visited and what is in the priority queue
    visited = []
    queue = [[mst_dis+distance_min(start_position, end_position), start_position, [start_position]]]
    seen = [] # mainly record purposes

    while queue:
        node = heapq.heappop(queue)
        path = node[2]
        (row, col) = (node[1][0], node[1][1])

        if node[1] not in seen:
            seen.append(node[1])

        if node[1] not in visited:
            visited.append(node[1])
            if (row, col) in end_position:
                if len(end_position) == 1:
                    return path+[node[1]], len(seen)
                else:
                    end_position.remove((row, col))
                    mst_dis = findMST(maze, end_position)
                    visited =[]
                    queue = [[mst_dis + distance_min(node[1], end_position), node[1], path]]

            neighbors = maze.getNeighbors(row, col)
            for i, neighbor in enumerate(neighbors):
                position_queue = [x[1] for x in queue]
                if neighbor not in position_queue:
                    heapq.heappush(queue,[mst_dis + distance_min(neighbor, end_position)+ len(path) + 1, neighbor, path + [neighbor]])
                else:
                    index = position_queue.index(neighbor)
                    if (mst_dis + distance_min(neighbor, end_position)) + len(path) + 1 < queue[index][0]:
                        heapq.heappush(queue, [mst_dis + distance_min(neighbor, end_position) + len(path) + 1, neighbor, path + [neighbor]])

    return

def astarsingle(maze, start_position, end_position):
    num_states_explored =0
    # book keeping, which node is visited and what is in the priority queue
    visited, queue = [], [(distance(start_position, end_position), start_position, 0)]
    parent_map = {start_position: start_position}
    not_found = True
    while queue and not_found:
        num_states_explored += 1
        node = heapq.heappop(queue)
        vertex = node[1]
        visited.append(vertex)
        if (vertex[0], vertex[1]) == end_position:
            not_found = False
            break
        for neighbour in maze.getNeighbors(vertex[0], vertex[1]):
            # if neighbour not in visited and neighbour not in (x[1] for x in queue):
            if neighbour not in visited:
                position_queue = [x[1] for x in queue]
                if neighbour not in position_queue:
                    heapq.heappush(queue, (distance(neighbour, end_position)+node[2]+1, neighbour, node[2]+1))
                    parent_map[neighbour] = (vertex[0], vertex[1])
                else:
                    index = position_queue.index(neighbour)
                    if (distance(neighbour, end_position)+node[2]+1) < queue[index][0]:
                        heapq.heappush(queue,
                                       (distance(neighbour, end_position) + node[2] + 1, neighbour, node[2] + 1))
                        parent_map[neighbour] = (vertex[0], vertex[1])

    # backtrack to find the trace
    parent_list = list()
    current = end_position
    while current != start_position:
        parent_list.append(current)
        current = parent_map[current]
    parent_list.append(current)
    parent_list.reverse()
    return parent_list, num_states_explored

def distance(start_pos, end_pos):
    dis = abs(end_pos[0] - start_pos[0]) + abs(end_pos[1] - start_pos[1])
    return dis

def distance_min(current, end_pos):

    max = -1
    maxindex=0
    min = -1
    minindex = 0
    for i, item in enumerate(end_pos):
        currentdis = distance(current, item)
        if currentdis > max or max == -1:
            maxindex=i
            max = currentdis
        if currentdis < min or min == -1:
            minindex=i
            min = currentdis


    return  min

def findMST(maze,end_position):
    end_list = list(itertools.combinations(range(len(end_position)), 2))
    for i, pairs in enumerate(end_list):
        (parent_list, num_states_explored) = astarsingle(maze, end_position[pairs[0]], end_position[pairs[1]])
        end_list[i] = [pairs[0], pairs[1], len(parent_list)]

    g = Graph(len(end_position))
    for i, pairs in enumerate(end_list):
        g.addEdge(pairs[0], pairs[1], pairs[2])

    sum =0
    result = g.KruskalMST()
    for i in result:
        sum =sum + i[2]
    return sum

# This class is adopted from internet, https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/
# Python program for Kruskal's algorithm to find
# Minimum Spanning Tree of a given connected,
# undirected and weighted graph

# Class to represent a graph
class Graph:

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []  # default dictionary
        # to store graph

    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    # A utility function to find set of an element i
    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    # A function that does union of two sets of x and y
    # (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    # The main function to construct MST using Kruskal's
    # algorithm
    def KruskalMST(self):

        result = []  # This will store the resultant MST

        i = 0  # An index variable, used for sorted edges
        e = 0  # An index variable, used for result[]

        # Step 1:  Sort all the edges in non-decreasing
        # order of their
        # weight.  If we are not allowed to change the
        # given graph, we can create a copy of graph
        self.graph = sorted(self.graph, key=lambda item: item[2])

        parent = []
        rank = []

        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        # Number of edges to be taken is equal to V-1
        while e < self.V - 1:

            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            # If including this edge does't cause cycle,
            # include it in result and increment the index
            # of result for next edge
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
                # Else discard the edge

        return  result