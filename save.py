

def bfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored

    if (len(maze.getObjectives()) >= 1):
        return bfsMultiple(maze)

    visited = []
    start = maze.getStart()
    # Create a queue whose element is a tuple,
    # each of which consists of a position point
    # and a path as a list of positions
    q = queue.Queue()
    q.put((start, [start]))
    while not q.empty():
        # Get the current position and path from the queue
        (current, path) = q.get()
        (row, col) = (current[0], current[1])
        # Check if the current position was visited
        if current not in visited:
            visited.append(current)
            # Check if reach the goal
            if maze.isObjective(row, col):
                return path, len(visited)
            # Add neighboring positions to the queue
            neighbors = []
            neighbors = maze.getNeighbors(row, col)
            for i, neighbor in enumerate(neighbors):
                q.put((neighbor, path + [neighbor]))
    return [], 0


def bfsMultiple(maze):
    # TODO: Write your code here
    # return path, num_states_explored

    global globalpath
    global globallen

    globalpath = []
    globallen = 0

    visited = []
    start = maze.getStart()
    goalPoints = maze.getObjectives()
    length = 0
    path = []

    bfsMultiplehelper(maze, start, goalPoints, visited, length, path)

    return globalpath, globallen


def bfsMultiplehelper(maze, start, goalPoints, visited, oldlength, oldpath):
    q = queue.Queue()
    q.put((start, [start]))
    while not q.empty():
        # Get the current position and path from the queue
        (current, path) = q.get()
        (row, col) = (current[0], current[1])
        # Check if the current position was visited
        if current not in visited:
            visited.append(current)
            # Check if reach the dot
            if (row, col) in goalPoints:

                if len(goalPoints) == 1:
                    global globalpath
                    global globallen

                    # print("old len:")
                    # print(len(oldpath + path))
                    # print("best len:")
                    # print(len(globalpath))

                    if len(oldpath + path) < len(globalpath) or globalpath == []:
                        globalpath = oldpath + path
                        globallen = len(visited) + oldlength
                        print("newlength!!!")
                        print(len(globalpath))
                        return
                else:
                    newvisited = []
                    newstart = current
                    newgoalPoints = copy.copy(goalPoints)
                    newgoalPoints.remove((row, col))
                    # print("new:")
                    # print(newgoalPoints)
                    bfsMultiplehelper(maze, newstart, newgoalPoints, newvisited, len(visited) + oldlength,
                                      oldpath + path)

            # Add neighboring positions to the queue
            neighbors = []
            neighbors = maze.getNeighbors(row, col)
            for i, neighbor in enumerate(neighbors):
                q.put((neighbor, path + [neighbor]))
    return