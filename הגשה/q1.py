# Import Libraries:

import copy
import math
import queue


# Functions :

def find_path(starting_board, goal_board, search_method, detail_output):  # Main function.
    # Lists of Agents (Kings & Bishops) and Force Fields:
    forceField_Array = []  # List for Force Fields coordinates.
    start_Agents = []  # List for Agents in the starting board coordinates.
    goal_Agents = []  # List for Agents in the goal board coordinates.

    for row in range(0, 6):
        for col in range(0, 6):

            # Starting Board:
            if starting_board[row][col] == 1:  # 1 -> Force Field =  "@", Stays constant.
                forceField_Array.append([row, col])
            elif starting_board[row][col] == 2:  # 2 -> King Agent = "*".
                start_Agents.append([row, col, '*'])  # [row, col, agent type].
            elif starting_board[row][col] == 3:  # 3 -> Bishop Agent = "&".
                start_Agents.append([row, col, '&'])

            # Goal Board:
            if goal_board[row][col] == 2:  # 2 -> King Agent = "*".
                goal_Agents.append([row, col, '*'])
            elif goal_board[row][col] == 3:  # 3 -> Bishop Agent = "&".
                goal_Agents.append([row, col, '&'])

    # Call for search method:
    if search_method == 1:
        astar_search(start_Agents, forceField_Array, goal_Agents, detail_output)


def astar_search(start_Agents, forceField_Array, goal_Agents, detail_output):
    optionsList = createPriority()  # PriorityQueue().
    visitedList = []

    h_Start_King = calculateHeuristic(getSpecificAgents(start_Agents, "*"),
                                      getSpecificAgents(copy.deepcopy(goal_Agents), "*"),
                                      1)  # 1 = Manhattan Distance.
    h_Start_Bish = calculateHeuristic(getSpecificAgents(start_Agents, "&"),
                                      getSpecificAgents(copy.deepcopy(goal_Agents), "&"),
                                      2)  # 2 = Hamming Distance + Euclidean Distance.

    h_Start_State = h_Start_King + h_Start_Bish

    startTupleState = (h_Start_State + 0, h_Start_State, 0, start_Agents, None)  # (f,h,g,Initial Board, the agent that he came from - his "Father")
    insertPriority(optionsList, startTupleState)  # PriorityQueue().put() , inserting the first option of the start Board.
    while not optionsList.empty():  # Checking all the options, if we didn't find a solution we will print "No path found." .
        currentState = optionsList.get()  # Pops the first path each time with the MIN f value.
        # currentState = startTupleState.
        # currentState[0] = f
        # currentState[1] = h
        # currentState[2] = g
        # currentState[3] = location array of agents - list of lists[[],[]]
        # currentState[4] = Father locations.
        if isGameFinished(currentState[3], goal_Agents) is True:  # Check if we found a solution (if currentState[3] - the current board is equivalent to goal board).
            getPath(currentState, forceField_Array, detail_output)  # currentState = current path, if is true we finished the game, and we will get the final path.
            return

        neighbors_Location = getNeighbors(currentState[3],
                                          forceField_Array)  # List of lists ,Neighbors, finds all valid neighbors to current state.
        for neighbor in neighbors_Location:  # for each neighbor in currentState[3] location array.
            if isInListCheck(visitedList,
                             neighbor):  # if True -> Continue, Add to visitedList - to avoid cases when state is already visited.
                continue
            if isInListCheck(optionsList,
                             neighbor):  # Avoid cases when state is already "Checked" in optionsList (in the queue) - for reducing the complexity time.
                continue
            # Calculate the A Star parameters:
            h_King = calculateHeuristic(getSpecificAgents(neighbor, "*"),
                                        getSpecificAgents(copy.deepcopy(goal_Agents), "*"),
                                        1)  # 1 = Manhattan Distance.
            h_Bish = calculateHeuristic(getSpecificAgents(neighbor, "&"),
                                        getSpecificAgents(copy.deepcopy(goal_Agents), "&"),
                                        2)  # 2 = Hamming Distance + Euclidean Distance.

            h = h_King + h_Bish
            g = currentState[2] + 1  # currentState[2] = g
            f = h + g
            tuple_to_queue = (f, h, g, neighbor, currentState)
            insertPriority(optionsList, tuple_to_queue)  # Add states\options to the optionsList.

        visitedList.append(currentState[3])  # currentState[3] = agent locations array, that we VISITED.

    print("No path found.")


def calculateHeuristic(currentStateLocs, goalStateLocs, heuristicCal):
    h = 0

    # Check :
    if (len(currentStateLocs) == 0) and (len(goalStateLocs) == 0):  # if both empty, there is not any Agent Type.
        return 0
    diffAgents = len(currentStateLocs) - len(goalStateLocs)  # Check if there is same number of Agents in both boards.
    if len(currentStateLocs) == 0 or diffAgents < 0:
        return math.inf  # To handle cases when there are no Agents in the board or there are fewer Agents in start board then goal board.
    if diffAgents > 0:
        for agent in currentStateLocs:
            h += 2 * (6 - agent[0])  # Giving more "Weight" to agents that are excess in current board compared to goal board.

    for agent in currentStateLocs:
        heuristicValArray = []
        if len(goalStateLocs) == 0:  # When there are no agents left in goal board.
            break
        for goalAgent in goalStateLocs:
            if heuristicCal == 1:  # Manhattan Distance.
                heuristicValArray.append(manhattanDistance(agent, goalAgent))  # Calculating manhattan dist from one agent to all agents in goal board.
            if heuristicCal == 2:  # Hamming Distance + Euclidean Distance.
                if agent not in goalStateLocs:
                    h = h + 1
                heuristicValArray.append(h + euclideanDistance(agent, goalAgent))

        minimum = math.inf  # We want the H value will be always the minimum cost path for each selected agent.
        min_index = 0
        for i in range(len(heuristicValArray)):
            if heuristicValArray[i] < minimum:
                minimum = heuristicValArray[i]
                min_index = i
        h = h + minimum
        goalStateLocs.pop(min_index)  # For each time we compare our current agent to goal anent we remove it from list.

    return h


def getSpecificAgents(agentsArray, agenType):  # Classified the array by agent types.
    specificArray = []
    for agent in agentsArray:
        if agent[2] == agenType:
            specificArray.append(agent)
    return specificArray


def euclideanDistance(currAgent, goalAgent):
    costDist = 1  # The cost for each step is 1.
    d_x = pow(currAgent[0] - goalAgent[0], 2)  # Can Only Walk Up\Down & Forward\Backward.
    d_y = pow(currAgent[1] - goalAgent[1], 2)
    return int(costDist * math.sqrt(math.pow(d_x, 2) + math.pow(d_y, 2)))


def manhattanDistance(currAgent, goalAgent):  # Using "Manhattan Distance" for calculate the Heuristic distance "h".
    costDist = 1  # The cost for each step is 1.
    d_x = abs(currAgent[0] - goalAgent[0])  # Can Only Walk Up\Down & Forward\Backward.
    d_y = abs(currAgent[1] - goalAgent[1])
    return costDist * (d_x + d_y)  # {(X axes distance) + (Y axes distance)} * Cost per step.


def createPriority():
    return queue.PriorityQueue()  # The lowest valued entries are retrieved first,
    # the lowest valued entry is the one returned by sorted(list(entries))[0]).
    # A typical pattern for entries is a tuple in the form: (priority_number, data).

def insertPriority(optionsList, s):
    optionsList.put(s)

def getPath(curr, forceField_Array, detail_output):  # Get the path. curr =
    stateList = []  # to restore the route of the agent.
    heuristicValList = []
    stateList.append(curr[3])  # adding the curr state to route
    heuristicValList.append(curr[1])
    pointer = curr[4]  # stores the last state that curr visited at

    while pointer[4] is not None:  # there is no pointer to the first state , pointer[4] = stateTuple.
        stateList.append(pointer[3])  # pointer[3] = current locations state of agents.
        heuristicValList.append(pointer[1])  # pointer[1] = h.
        pointer = pointer[4]  # points to last of last state(pinter holds tuple).

    stateList.append(pointer[3])
    heuristicValList.append(pointer[1])
    stateList.reverse()  # Reverses the list from end to the beginning.
    heuristicValList.reverse()

    for i in range(len(stateList)):
        printBoardGame(stateList[i], forceField_Array, i + 1, len(stateList))  # prints the route
        if detail_output is True:
            print("Heuristic: " + str(heuristicValList[i]))  # prints the heuristic if it is needed
            print('------')
        else:
            print('------')

def isGameFinished(start_Agents, goal_Agents):  # Check if all the Agents stand in the right positions -> End Game.
    for agent in start_Agents:
        if agent not in goal_Agents:
            return False
    if len(start_Agents) == len(goal_Agents):  # if the last check is True - checking if there are no Unnecessary agents on board.
        return True
    return False


def getNeighbors(currentStateLocs, forceField_Array):  # finding neighbors (that are located in legal places)
    neighbors = []
    index = 0

    for agent in currentStateLocs:  # Array of agents locations, "For each location" , agent = Agent Location.
        if agent[2] == "*":  # King Walk
            for row in range(-1, 1):
                for col in range(-1, 2):
                    if isNextMoveOK([agent[0] + row, agent[1] + col, agent[2]], forceField_Array, currentStateLocs):
                        resX = copy.deepcopy(currentStateLocs)
                        resX[index][0] = resX[index][0] + row  # Row
                        resX[index][1] = resX[index][1] + col  # Col
                        neighbors.append(resX)
            for col in range(-1, 2):
                row = 1
                if agent[0] + row == 6:  # To get rid of agents that crosses the 6'th row.
                    res = copy.deepcopy(currentStateLocs)
                    res.remove(agent)
                    neighbors.append(res)  # List of removed agents from board.
                else:
                    if isNextMoveOK([agent[0] + row, agent[1] + col, agent[2]], forceField_Array, currentStateLocs):
                        resX = copy.deepcopy(currentStateLocs)
                        resX[index][0] = resX[index][0] + row  # Row
                        resX[index][1] = resX[index][1] + col  # Col
                        neighbors.append(resX)
        # Bishop Walk:
        if agent[2] == "&":
            bishop = agent
            for DL in range(1, 6):
                # Check Down and Left, "Row +1 and Col -1":
                if isNextMoveOK([bishop[0] + DL, bishop[1] - DL, bishop[2]], forceField_Array, currentStateLocs):
                    if bishop[0] + DL == 6:  # Agents after the 6'th row disappear.
                        resDL = copy.deepcopy(currentStateLocs)
                        resDL.remove(bishop)
                        neighbors.append(resDL)  # List of removed agents from board.
                    else:
                        resDownLeft = copy.deepcopy(currentStateLocs)
                        resDownLeft[index][0] = resDownLeft[index][0] + DL  # Row
                        resDownLeft[index][1] = resDownLeft[index][1] - DL  # Col
                        neighbors.append(resDownLeft)
                else:
                    break
            for DR in range(1, 6):
                # Check Down and Right, "Row +1 and Col +1":
                if isNextMoveOK([bishop[0] + DR, bishop[1] + DR, bishop[2]], forceField_Array, currentStateLocs):
                    if bishop[0] + DR == 6:  # Agents after the 6'th row disappear.
                        resDR = copy.deepcopy(currentStateLocs)
                        resDR.remove(bishop)
                        neighbors.append(resDR)  # List of removed agents from board.
                    else:
                        resDownRight = copy.deepcopy(currentStateLocs)
                        resDownRight[index][0] = resDownRight[index][0] + DR  # Row
                        resDownRight[index][1] = resDownRight[index][1] + DR  # Col
                        neighbors.append(resDownRight)
                else:
                    break
            for UR in range(1, 6):
                # Check Up and Right, "Row -i and Col +i":
                if isNextMoveOK([bishop[0] - UR, bishop[1] + UR, bishop[2]], forceField_Array, currentStateLocs):
                    resUpRight = copy.deepcopy(currentStateLocs)
                    resUpRight[index][0] = resUpRight[index][0] - UR  # Row
                    resUpRight[index][1] = resUpRight[index][1] + UR  # Col
                    neighbors.append(resUpRight)
                else:
                    break
            for UL in range(1, 6):
                # Check Up and Left, "Row -1 and Col -1":
                if isNextMoveOK([bishop[0] - UL, bishop[1] - UL, bishop[2]], forceField_Array, currentStateLocs):
                    resUpLeft = copy.deepcopy(currentStateLocs)
                    resUpLeft[index][0] = resUpLeft[index][0] - UL  # Row
                    resUpLeft[index][1] = resUpLeft[index][1] - UL  # Col
                    neighbors.append(resUpLeft)
                else:
                    break
        index += 1
    return neighbors

def isNextMoveOK(nextLocation, forceField_Array, currentState):  # Check if the next move is a legal move.

    if nextLocation[0:2] in forceField_Array:  # Check for Force Field , agents cant step on a force field.
        return False
    elif (nextLocation[0] < 0) or (nextLocation[1] < 0) or (nextLocation[1] > 5) or (nextLocation[0] > 6):  # Board boundaries check.
        return False
    for currAgent in currentState:
        if nextLocation[0:2] == currAgent[0:2]:  # Check if the nextLocation stepping on another agent (King/Bishop).
            return False
    else:
        return True

def isInListCheck(list, var):  # This function is to check if the neighbor is already checked.
    # list = List of location lists var -> [ [[],[]], [[],[]] ]
    # var = List of locations -> [[],[]]
    # Priority queue :
    if isinstance(list, queue.PriorityQueue):
        if list.qsize() != 0:
            for i in list.queue:
                count = 0
                for j in var:
                    if j in i[3]:
                        count += 1
                if count == len(i[3]) and count == len(
                        var):  # if the amount of agents in neighbors and in optionsList equivalent to len of the similarities.
                    return True
            return False
        return False
    # Regular list
    for i in list:
        count = 0
        for j in var:
            if j in i:
                count += 1

        if count == len(i) and count == len(var):
            return True
    return False

def printBoardGame(state, forceField_Array, index, len):  # Board print function.

    if index == 1:
        print('\nBoard 1 (starting position):')
    elif index == len:
        print('Board ' + str(index) + ' (goal position):')
    else:
        print('Board ' + str(index) + ':')
    print('  1 2 3 4 5 6')
    for row in range(6):
        print(row + 1, end='')
        for col in range(6):
            if [row, col, "*"] in state:
                print(' *', end='')
            elif [row, col, "&"] in state:
                print(' &', end='')
            elif [row, col] in forceField_Array:
                print(' @', end='')
            else:
                print('  ', end='')
        print()

A = [[2, 0, 2, 0, 3, 0],
     [0, 0, 0, 2, 1, 3],
     [1, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 1, 0],
     [3, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0]]

B = [[0, 0, 0, 2, 3, 0],
     [0, 2, 0, 0, 1, 3],
     [1, 0, 3, 0, 2, 0],
     [0, 0, 1, 0, 1, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0]]

find_path(A, B, 1, True)