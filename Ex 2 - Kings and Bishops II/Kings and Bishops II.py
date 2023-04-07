# Import Libraries:

import copy
import math
import queue
import random
import numpy as np
import sys


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
    if search_method == 2:
        hill_Climbing(start_Agents, forceField_Array, goal_Agents, detail_output)
    if search_method == 3:
        simulated_Annealing(start_Agents, forceField_Array, goal_Agents, detail_output)
    if search_method == 4:
        k_Beam(start_Agents, forceField_Array, goal_Agents, detail_output)
    if search_method == 5:
        genetic_Algorithm(start_Agents, forceField_Array, goal_Agents, detail_output)

def optFitness(population, goal_Agents):
    state_and_heuristic_list = []
    pop_with_score = []
    # Create a list of [[state, heuristic],[state, heuristic],..]
    for state in population:
        h_King = calculateHeuristic(getSpecificAgents(state[0], "*"),
                                          getSpecificAgents(copy.deepcopy(goal_Agents), "*"),
                                          1)  # 1 = Manhattan Distance.
        h_Bish = calculateHeuristic(getSpecificAgents(state[0], "&"),
                                          getSpecificAgents(copy.deepcopy(goal_Agents), "&"),
                                          2)  # 2 = Hamming Distance + Euclidean Distance.
        h = h_King + h_Bish
        state_and_heuristic_list.append([state, h])  # calculates the regular heuristic
    # Calculates sum and max of the heuristics
    sum_heuristics = sum([h[1] for h in state_and_heuristic_list if h[1] is not math.inf])
    # Returns list of [[state, fitness score],[state, fitness score],...]
    for state in state_and_heuristic_list:
        if (state[1] is not math.inf) and (len(state[0][0]) > 0):
            pop_with_score.append([state[0], (sum_heuristics - state[1]) / sum_heuristics])
        else:
            continue
    sum_normalized_heuristics = sum([h[1] for h in pop_with_score])
    # Normalize.
    for state in pop_with_score:
        state[1] = (state[1] / sum_normalized_heuristics)
    return pop_with_score

def reProduce(parent_1, parent_2):
    P1_Kings = getSpecificAgents(parent_1, "*")
    P2_Kings = getSpecificAgents(parent_2, "*")
    P1_Bishops = getSpecificAgents(parent_1, "&")
    P2_Bishops = getSpecificAgents(parent_2, "&")

    lenK = max(len(P1_Kings), len(P2_Kings))
    lenB = max(len(P1_Bishops), len(P2_Bishops))

    kings = P1_Kings + P2_Kings
    bishops = P1_Bishops + P2_Bishops
    child_K = []
    child_B = []

    while len(child_K) != lenK:
        if P1_Kings == P2_Kings:
            child_K = P1_Kings
            continue
        Ck = random.choice(kings)
        if Ck not in child_K:
            child_K.append(Ck)
            kings.remove(Ck)
    while len(child_B) != lenB:
        if P1_Bishops == P2_Bishops:
            child_B = P1_Bishops
            continue
        Cb = random.choice(bishops)
        if Cb not in child_B:
            child_B.append(Cb)
            bishops.remove(Cb)

    child = child_K + child_B
    return child

def isInGen(lst1, query):
    # Adapting "isInListCheck" function to the Genetic Algorithm.
    new_lst = [state[0][0] for state in lst1]
    return isInListCheck(copy.deepcopy(new_lst), query)

def childMutate(child, forceField_Array):
    mutateOptions = getNeighbors(child, forceField_Array)
    return random.choice(mutateOptions)

def getPath_GeneticAlgo(curr, forceField_Array):
    stateList = []
    # Appends the goal state
    stateList.append(curr[0])
    pointer = curr[2]
    #print(curr)
    while pointer[2] is not None:
        stateList.append(pointer[0])
        pointer = pointer[2]
    # Appends the root
    stateList.append(pointer[0])
    stateList.reverse()
    for i in range(len(stateList)):
        printBoardGame(stateList[i], forceField_Array, i + 1, len(stateList))
        print("------")

def genetic_Algorithm(start_Agents, forceField_Array, goal_Agents, detail_output):
    sys.setrecursionlimit(10000)  # for the copy.deepcopy(x) limitations in space
    pop_with_score = []
    details = []
    current_state = [start_Agents, None, None,  None]   # I save each state at [current, mutant, parent1, parent2] form to help me restore the route
    current_population = getNeighbors(current_state[0], forceField_Array)
    current_population = current_population[:10]
    current_population = list(map(lambda x: [x, None, current_state, current_state], current_population))  # stores each initial board neig in the same form
    pop_with_score = optFitness(current_population, goal_Agents)
    pop_with_score.sort(key=lambda N: N[1], reverse=True)
    done = False  # if done stop to fill the list
    blocked = False  # if one of agent is blocked - stop search legal moves
    if len(current_population) == 0:
        blocked = True
    # limit generations that produced
    for Xi in range(150):
        if blocked:
            break
        next_population = []
        iteration = 0
        while len(next_population) < 10:
            while True:  # Run until we found 2 parents.
                # Parent 1:
                randChoose1 = round(random.uniform(0, 4))
                P1 = pop_with_score[randChoose1]
                parent_1 = P1[0][0]
                parent_1_score = P1[1]
                # Parent 2:
                randChoose2 = round(random.uniform(0, 4))
                P2 = pop_with_score[randChoose2]
                parent_2_score = P2[1]
                parent_2 = P2[0][0]
                random.seed()
                if len(parent_1) == len(parent_2):
                    child = reProduce(parent_1, parent_2)
                    child = [child, None, P1[0], P2[0]]
                    iteration += 1
                    break
            if len(child[0]) > 0:
                random_num = random.random()
                # Probability to have a mutation -> this will "Forward" the process.
                if random_num < 0.5:
                    mutant = childMutate(child[0], forceField_Array)
                    child = [mutant, child, child[2], child[3]]
            if isGameFinished(child[0], goal_Agents):
                getPath_GeneticAlgo(child, forceField_Array)
                # print for detail outPut == True
                if detail_output is True and len(details) > 0:
                    child_state = details[0][0]
                    p1_state = details[0][2][0]
                    p2_state = details[0][3][0]
                    p1_score = details[1]
                    p2_score = details[2]
                    if details[0][1] is not None:
                        mutate_happened = details[0][1][0]
                    else:
                        mutate_happened = None
                    print("Starting board 1 (probability of selection from population:" + str(round(p1_score, 5)) + "):")
                    printBoardGame(p1_state, forceField_Array, None, 1)
                    print('------')
                    print("Starting board 2 (probability of selection from population:" + str(round(p2_score, 5)) + "):")
                    printBoardGame(p2_state, forceField_Array, None, 1)
                    print('------')
                    if mutate_happened is not None:
                        print("Result board (mutation happened: Yes):")
                        printBoardGame(child_state, forceField_Array, None, 1)
                    else:
                        print("Result board (mutation happened: No):")
                        printBoardGame(child_state, forceField_Array, None, 1)
                done = True
                break
            # Help condition for printing when detail output is True.
            if iteration == 1 and len(next_population) == 0:
                details.append(child)
                details.append(parent_1_score)
                details.append(parent_2_score)
            next_population.append(child)
        if done:
            break
        current_population = next_population
        pop_with_score = optFitness(current_population, goal_Agents)
        pop_with_score.sort(key=lambda N: N[1], reverse=True)
    if not done:
        print("No path found.")

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
            getPath(currentState, forceField_Array, detail_output, 1, None)  # currentState = current path, if is true we finished the game, and we will get the final path.
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

def hill_Climbing(start_Agents, forceField_Array, goal_Agents, detail_output):
    # Initials:
    h_Initial_King = calculateHeuristic(getSpecificAgents(start_Agents, "*"),
                                      getSpecificAgents(copy.deepcopy(goal_Agents), "*"),
                                      1)  # 1 = Manhattan Distance.
    h_Initial_Bish = calculateHeuristic(getSpecificAgents(start_Agents, "&"),
                                      getSpecificAgents(copy.deepcopy(goal_Agents), "&"),
                                      2)  # 2 = Hamming Distance + Euclidean Distance.
    h_initial = h_Initial_King + h_Initial_Bish
    initial_Neighbors = getNeighbors(start_Agents, forceField_Array)  # Finds all valid neighbors for the initial state.
    iteration = 0  # Iteration counter , 5 iterations in total.
    isFinished = False  # Boolean for breaking inner loop.
    initialState = [None, h_initial, None, start_Agents, None]

    # initialState[0] = f -> None (fake values to fit the first exercise).
    # initialState[1] = h
    # initialState[2] = g -> None (fake values to fit he first exercise).
    # initialState[3] = The current state locations & agents.
    # initialState[4] = Father locations.

    while iteration < 5:  # 5 Random restarts.
        if isFinished:
            break
        h_CurrentSolution = h_initial  # Reset the value to the initial board each iteration.

        # Iterations 1-5 choose random neighbor from start:
        if iteration != 0:
            if len(initial_Neighbors) > 0:
                rand_Index = random.randrange(len(initial_Neighbors))  # Choose neighbor randomly (only after initial iteration).
                # if rand_Index not in visited_list:
                next_StateNeighbors = [None, h_CurrentSolution, None, initial_Neighbors[rand_Index], initialState]  # in the last index I save the "Father".
                initial_Neighbors.pop(rand_Index)
            else:
                break

        # Only first iteration chooses the best neighbor from start.
        else:
            HC_HeuristicNeighbors = []
            for i in range(len(initial_Neighbors)):     # Add H value to each neighbor.
                h_King = calculateHeuristic(getSpecificAgents(initial_Neighbors[i], "*"),
                                                  getSpecificAgents(copy.deepcopy(goal_Agents), "*"),
                                                  1)  # 1 = Manhattan Distance.
                h_Bish = calculateHeuristic(getSpecificAgents(initial_Neighbors[i], "&"),
                                                  getSpecificAgents(copy.deepcopy(goal_Agents), "&"),
                                                  2)  # 2 = Hamming Distance + Euclidean Distance.
                h_HC = h_King + h_Bish
                HC_HeuristicNeighbors.append([None, h_HC, None, initial_Neighbors[i], initialState])

            if len(HC_HeuristicNeighbors) > 0 and getBestNeighbor(HC_HeuristicNeighbors)[1] <= h_initial:   # if the heuristic value is bigger, Break !.
                next_StateNeighbors = getBestNeighbor(HC_HeuristicNeighbors)  # Get the next state with minimum heuristic.
                initial_Neighbors.pop(initial_Neighbors.index(next_StateNeighbors[3]))  # Removes from initial_Neighbors to avoid cases that there is repetition of stats from goal board.
            else:
                break

        while True:     # if TRUE -> searching
            if isGameFinished(next_StateNeighbors[3], goal_Agents) is True:  # Checks compatibility between the current state and the goal state.
                isFinished = True
                getPath(next_StateNeighbors, forceField_Array, detail_output, 2, None)  # Prints the path to the goal state (when found).
                break

            if len(next_StateNeighbors[3]) == 0:  # if the state is an empty state then move to another.
                break

            inner_Neighbors = getNeighbors(next_StateNeighbors[3], forceField_Array)
            HC_HeuristicNeighbors_inner = []
            fatherhood_State = next_StateNeighbors  # Saves the state that current came from.
            for neighbor in inner_Neighbors:
                h_King2 = calculateHeuristic(getSpecificAgents(neighbor, "*"),
                                                  getSpecificAgents(copy.deepcopy(goal_Agents), "*"),
                                                  1)  # 1 = Manhattan Distance.
                h_Bish2 = calculateHeuristic(getSpecificAgents(neighbor, "&"),
                                                  getSpecificAgents(copy.deepcopy(goal_Agents), "&"),
                                                  2)  # 2 = Hamming Distance + Euclidean Distance.
                h_HC2 = h_King2 + h_Bish2
                HC_HeuristicNeighbors_inner.append([None, h_HC2, None, neighbor, fatherhood_State])

            if getBestNeighbor(HC_HeuristicNeighbors_inner)[1] <= h_CurrentSolution:    # if the heuristic value is bigger, Break !.
                next_StateNeighbors = getBestNeighbor(HC_HeuristicNeighbors_inner)  # get the next state with minimum heuristic
                h_CurrentSolution = getBestNeighbor(HC_HeuristicNeighbors_inner)[1]  # saves the current heuristic for choosing the next one
            else:
                break
        iteration += 1
    if not isFinished:
        print("No path found")

def getBestNeighbor(heuristic_and_agents_list):
    # Get the neighbor that returns the minimum value.
    min_index = 0
    minimum = math.inf
    for i in range(len(heuristic_and_agents_list)):
        if heuristic_and_agents_list[i][1] < minimum:
            minimum = heuristic_and_agents_list[i][1]
            min_index = i
    res = heuristic_and_agents_list[min_index]
    return res

def simulated_Annealing(start_Agents, forceField_Array, goal_Agents, detail_output):
    # ---- Initials: ----
    visitedList = []
    list_of_states_and_prob = []
    iteration = 0

    h_Initial_King = calculateHeuristic(getSpecificAgents(start_Agents, "*"),
                                      getSpecificAgents(copy.deepcopy(goal_Agents), "*"),
                                      1)  # 1 = Manhattan Distance.
    h_Initial_Bish = calculateHeuristic(getSpecificAgents(start_Agents, "&"),
                                      getSpecificAgents(copy.deepcopy(goal_Agents), "&"),
                                      2)  # 2 = Hamming Distance + Euclidean Distance.
    h_initial = h_Initial_King + h_Initial_Bish

    currentState = [None, h_initial, None, copy.deepcopy(start_Agents), None, []]

    for t in range(100):    # ( No longer than t = 100 )
        T = scheduleT(t)
        random.seed()

        if isGameFinished(currentState[3], goal_Agents):
            getPath(currentState, forceField_Array, detail_output, 3, None)
            break
        # Find all the Neighbors\Options of current state.

        neighborOptions = getNeighbors(currentState[3], forceField_Array)

        # When number of iterations completed, the search as finished.
        if T == 0 or len(neighborOptions) == 0 or t == 99:
            print("No path found")
            break
        randomChoose = random.choice(neighborOptions)
        if isInListCheck(visitedList, randomChoose):
            continue
        currentKingH = calculateHeuristic(getSpecificAgents(currentState[3], "*"),
                                            getSpecificAgents(copy.deepcopy(goal_Agents), "*"),
                                            1)  # 1 = Manhattan Distance.
        currentBishopH = calculateHeuristic(getSpecificAgents(currentState[3], "&"),
                                            getSpecificAgents(copy.deepcopy(goal_Agents), "&"),
                                            2)  # 2 = Hamming Distance + Euclidean Distance.
        nextKingH = calculateHeuristic(getSpecificAgents(randomChoose, "*"),
                                            getSpecificAgents(copy.deepcopy(goal_Agents), "*"),
                                            1)  # 1 = Manhattan Distance.
        nextBishopH = calculateHeuristic(getSpecificAgents(randomChoose, "&"),
                                            getSpecificAgents(copy.deepcopy(goal_Agents), "&"),
                                            2)  # 2 = Hamming Distance + Euclidean Distance.
        currentH = currentKingH + currentBishopH
        nextH = nextKingH + nextBishopH
        diff_E = currentH - nextH
        if diff_E <= 0:
            probability = np.exp(diff_E / T)
            rand = random.random()
            if rand > probability:  # LOSE
                currentState[5].append([currentState[3], randomChoose, round(probability, 5)])
            else:  # WIN - If we succeed to "Hit" the probability, we continue the progress.
                currentState[5].append([currentState[3], randomChoose, round(rand, 5)])
                currentState = [None, nextH, None, randomChoose, currentState, []]
                continue
        else:  # WIN
            X = [[currentState[3], randomChoose, 1]]
            currentState[5].append(X[0])
            currentState = [None, nextH, None, randomChoose, currentState, []]  # currentState[4] = Father
            continue

        visitedList.append(currentState[3])
        currentState[5].sort(key=lambda N: N[2])

def scheduleT(t):
    return 1 - ((1 / 100) * t)
    # Linear function that resets when t=0, the idea here is to give higher "weight" to initial boards.
    # The algorithm will take risks at the first boards and decrease it by time.

def getHeuristicNeighbors(open_list, forceField_Array, goal_Agents):
    heuristicNeighbors = []
    state = open_list.get()
    neighbors = getNeighbors(state[3], forceField_Array)
    for neighbor in neighbors:
        heuristic_King = calculateHeuristic(getSpecificAgents(neighbor, "*"),
                                            getSpecificAgents(copy.deepcopy(goal_Agents), "*"),
                                            1)  # 1 = Manhattan Distance.
        heuristic_Bish = calculateHeuristic(getSpecificAgents(neighbor, "&"),
                                            getSpecificAgents(copy.deepcopy(goal_Agents), "&"),
                                            2)  # 2 = Hamming Distance + Euclidean Distance.
        heuristic = heuristic_King + heuristic_Bish
        heuristicNeighbors.append([heuristic, heuristic, None, neighbor, state, None])  # adding to all neighbors list the value that removed from queue.
    return heuristicNeighbors

def k_Beam(start_Agents, forceField_Array, goal_Agents, detail_output):
    K = 3
    found = False
    visited = []  # all states that i visited at
    StateList = createPriority()
    h_Initial_King = calculateHeuristic(getSpecificAgents(start_Agents, "*"),
                                      getSpecificAgents(copy.deepcopy(goal_Agents), "*"),
                                      1)  # 1 = Manhattan Distance.
    h_Initial_Bish = calculateHeuristic(getSpecificAgents(start_Agents, "&"),
                                      getSpecificAgents(copy.deepcopy(goal_Agents), "&"),
                                      2)  # 2 = Hamming Distance + Euclidean Distance.
    h_initial = h_Initial_King + h_Initial_Bish
    current = [h_initial, h_initial, None, start_Agents, None, None]
    StateList.put(current)
    while not StateList.empty():
        kbNeighbors = getHeuristicNeighbors(StateList, forceField_Array, goal_Agents)
        kbNeighbors.sort(key=lambda N: N[1])  # function that sorts by heuristic the states
        kbNeighbors = kbNeighbors[:K]      # The K (K=3) Best Options.
        for neighbor in kbNeighbors:
            if isGameFinished(neighbor[3], goal_Agents):
                for N1 in kbNeighbors:
                    kActionList = []
                    while len(kbNeighbors) > 0:
                        pop_val = kbNeighbors.pop(0)
                        StateList.put(pop_val)
                        kActionList.append(pop_val[3])
                        visited.append(pop_val[3])
                        N1[5] = kActionList
                getPath(neighbor, forceField_Array, detail_output, 4, None)
                found = True
                break
        if found:
            break
        for N1 in kbNeighbors:
            kActionList = []
            while len(kbNeighbors) > 0:
                pop_val = kbNeighbors.pop(0)
                StateList.put(pop_val)
                kActionList.append(pop_val[3])
                visited.append(pop_val[3])
                N1[5] = kActionList
        for N2 in kbNeighbors:
            print(N2[5])

    if not found:
        print("No path found")

def getPath(curr, forceField_Array, detail_output, search_method, related_list):  # Get the path. curr =
    stateList = []  # to restore the route of the agent.
    heuristicValList = []
    kBeamList = []
    sAList = []

    stateList.append(curr[3])  # adding the curr state to route
    heuristicValList.append(curr[1])
    if search_method == 3:
        sAList.append(curr[5])
        print(curr[5])
    if search_method == 4:
        kBeamList.append(curr[5])

    pointer = curr[4]  # stores the last state that curr visited at

    while pointer[4] is not None:  # there is no pointer to the first state , pointer[4] = stateTuple.
        stateList.append(pointer[3])  # pointer[3] = current locations state of agents.
        heuristicValList.append(pointer[1])  # pointer[1] = h.
        if search_method == 4:
            kBeamList.append(pointer[5])
        if search_method == 3:
            sAList.append(pointer[5])
        pointer = pointer[4]  # points to last of last state(pinter holds tuple).
    stateList.append(pointer[3])
    heuristicValList.append(pointer[1])
    stateList.reverse()  # Reverses the list from end to the beginning.
    heuristicValList.reverse()
    if search_method == 3:
        sAList.append(pointer[5])
        sAList.reverse()
    if search_method == 4:
        kBeamList.append(pointer[5])
        kBeamList.reverse()

    if (search_method == 1) or (search_method == 2):
        for i in range(len(stateList)):
            printBoardGame(stateList[i], forceField_Array, i + 1, len(stateList))  # prints the route
            if detail_output is True:
                print("Heuristic: " + str(heuristicValList[i]))  # prints the heuristic if it is needed
                print('------')
            else:
                print('------')

    if search_method == 3:
        sAListC = copy.deepcopy(sAList)
        if len(sAListC[0]) == 0:
            sAListC = sAListC[1:]
        for i in range(len(stateList)):
            printBoardGame(stateList[i], forceField_Array, i + 1, len(stateList))  # prints the route
            if detail_output is True:
                if i == (len(stateList)-1):
                    break
                for R in sAListC[i]:
                    if getCoordinates(R) is None:
                        continue
                    if i == 0:
                        print("action: ", getCoordinates(R), "; probability: ", str(R[2]))     # prints the heuristic if it is needed
                print('------')
            else:
                print('------')

    if search_method == 4:  # K Beam Algorithm.
        for i in range(len(stateList)):
            if detail_output is True:
                if i != 1:
                    printBoardGame(stateList[i], forceField_Array, i + 1, len(stateList))
                    print('------')
                elif i == 1:
                    char = "a"
                    if kBeamList[i] is not None:
                        for opt in kBeamList[i]:
                            print("Board "+str(i+1)+str(char) + ':')
                            char = chr(ord(char) + 1)
                            printBoardGame(opt, forceField_Array, None, 3)
                            print('------')
                    printBoardGame(stateList[i], forceField_Array, i + 1, len(stateList))
            else:
                printBoardGame(stateList[i], forceField_Array, i + 1, len(stateList))
                print('------')

def getCoordinates(coordinatesProbList):
    currentLoc = copy.deepcopy(coordinatesProbList[0])
    nextLoc = copy.deepcopy(coordinatesProbList[1])
    for curr in coordinatesProbList[0]:
        for next in coordinatesProbList[1]:
            if (curr[0] == next[0]) and (curr[1] == next[1]):
                currentLoc.remove(curr)
                nextLoc.remove(next)

    if len(currentLoc) == len(nextLoc):
        return "(" + str(currentLoc[0][0]+1) + "," + str(currentLoc[0][1]+1) + ")->" + "(" + str(nextLoc[0][0]+1) + "," + str(nextLoc[0][1]+1) + ")"

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
        # print("\nIndex = ", index)
        # print("Current Location :", currentStateLocs)
        # print("Neighbors :\n")
        # for N in neighbors:
        #     print(N)
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
    elif index is None:
        print()
    else:
        print('Board ' + str(index) + ':')

    print('   1 2 3 4 5 6')
    for row in range(6):
        print(row + 1, end=':')
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

