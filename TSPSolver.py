#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import random
import copy

class Node:
	def __init__(self, lowerBound, matrix, pathSoFar, prevNode, thisNode):
		self.pathSoFar = pathSoFar #List of cities on the tour so far
		self.matrix = matrix.copy() #Matrix for the currently stored state
		self.lowerBound = lowerBound #Minimum possible cost for the branch
		self.leadingEdge = tuple((prevNode,thisNode)) #The tuple describing the edge to this node (prevNode,thisNode)

	def __lt__(self,other):
		return self.leadingEdge[1] < other.leadingEdge[1] #Overloaded less-than function

	def __eq__(self, other): #Overloaded equality function
		if(other == None):
			return False
		if(not isinstance(other, Node)):
			return False
		return self.leadingEdge[1] == other.leadingEdge[1]


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	def reducedCostMatrix(self,matrix):
		bound = 0
		resultMatrix = matrix
		for i in range(len(resultMatrix)): #For each row, subtract minimum value of the row from each cell of the row
			minimumVal = min(resultMatrix[i])

			if (minimumVal != float('inf')): #If the minimum is not infinity
				for k in range(len(resultMatrix[i])):
					resultMatrix[i][k] -= minimumVal
				bound += minimumVal

		for j in range(len(resultMatrix[0])): #For each column, subtract minimum value of column from each cell of the column
			minimumVal = float('inf')
			for k in range(len(resultMatrix)):
				if (resultMatrix[k][j] < minimumVal):
					minimumVal = resultMatrix[k][j]
			if (minimumVal != float('inf')): #If the minimum is not infinity
				for k in range(len(resultMatrix)):
					resultMatrix[k][j] -= minimumVal
				bound += minimumVal
			elif (j == 0): #If there is no path back to the start of the tour....
				print('Unique case reached')
				bound += float('inf') #....then the cost of the reduction can be set to infinity!

		return resultMatrix,bound 


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		
		results = {} #Result of the function
		cities = self._scenario.getCities()
		start_time = time.time()
		count = 0 #How many times the loop has iterated

		Solution = None
		iterator = 0 #Gives the while loop 5 chances to find a valid solution
		
		while (iterator <= 5):
			#visited = [False] * len(cities)
			count += 1
			route = []
			cost = 0 #FIXME (Maybe we'll need to use reduced cost matrices?)
			notVisited = [i for i in range(len(cities))]
			nextCity = random.choice(notVisited)
			firstCity = nextCity
			route.append(cities[nextCity])
			notVisited.remove(nextCity)
			for i in range(len(cities)-1):
				minDist = float('inf')
				startCity = nextCity		
				for j in range(len(notVisited)):
					dist = cities[startCity].costTo(cities[notVisited[j]])
					if dist < minDist:
						minDist = dist
						nextCity = notVisited[j]
				
				if minDist == float('inf'):
					iterator += 1
					break
				else:
					cost += minDist
					route.append(cities[nextCity])
					notVisited.remove(nextCity)
			if (len(route) == len(cities)): #End of tour
				if (cities[nextCity].costTo(cities[firstCity]) != float('inf')):
					cost += cities[nextCity].costTo(cities[firstCity])
					Solution = route
					break
				else:
					iterator += 1
		
		end_time = time.time()
		tspSolution = TSPSolution(route)
		results['cost'] = cost if Solution != None else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = tspSolution if Solution != None else None
		return results




	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=600.0 ):
		results = {} #Result of the function
		cities = self._scenario.getCities()
		start_time = time.time()
		pruned = 0 #Number of pruned branches
		theMax = 0 #Biggest the priority queue has gotten
		total = 0 #Number of total new states
		count = 0 #How many times the loop has iterated

		AdjMatrix = [[float('inf') for j in range(len(cities))] for i in range(len(cities))] #Initialize adjacency matrix

		for i in range(len(cities)): #Set up the distances of the adjacenty matrix
			for j in range(len(cities)):
				if i != j:			
					dist = cities[i].costTo(cities[j])
					AdjMatrix[i][j] = dist

		#AdjMatrix = [[float('inf'),7,3,12],[3,float('inf'),6,14],[5,8,float('inf'),6],[9,3,5,float('inf')]]
		#print(AdjMatrix)

		reducedCost = self.reducedCostMatrix(AdjMatrix) #Function to get reduced cost matrix
		AdjMatrix = reducedCost[0] #The first result of that function gives the matrix

		bound = reducedCost[1] #The second result of the function gives the reduction cost

		BSSF_Initial = float('inf')
		for i in range(5):
			aResult = self.defaultRandomTour(time_allowance)['cost']
			if aResult < BSSF_Initial:
				BSSF_Initial = aResult

		BSSF = BSSF_Initial #Get an initial BSSF
		#print('BSSF: ',BSSF)
		SolutionNodeSoFar = None #Stores the node containing the best tour that we know so far

		thisPath = [0] #Path that the tour is taking so far
		root = Node(bound,AdjMatrix,thisPath,None,0) #Starting node of the tour

		goodQueue = []
		heapq.heapify(goodQueue)
		#heapq.heappush(goodQueue, (tuple((len(AdjMatrix)-len(root.pathSoFar),bound)), root)) #Set depth and lower bound as key of priority queue
		key1 = tuple((len(AdjMatrix)-len(root.pathSoFar),bound))
		heapq.heappush(goodQueue, (key1, root))
		theMax = 1
		total = 1

		#WHEN TO PRUNE
		#==================================================================
		#Take a state off of the queue and it’s lb > bssf? +1 to pruned
		#New child that’s lb > bssf? +1 to the pruned
		while (len(goodQueue) != 0):
			count += 1 

			currentTime = time.time() - start_time
			if (currentTime > time_allowance):
				break #If exceeded time allotment, break out of the loop

			thisNode = heapq.heappop(goodQueue)[1] #Pop new state off of the queue

			if (thisNode.lowerBound > BSSF): #If this new node exceeds BSSF, then prune the branch
				pruned += 1
				continue
		
			children = [q for q in range(len(AdjMatrix))]
			iterator = 0
			#After this while loop finishes executing, we will have a list of all children states that haven't been visited yet
			while(len(children) != (len(AdjMatrix) - len(thisNode.pathSoFar) ) ):
				if (children[iterator] in thisNode.pathSoFar):
					children.remove(children[iterator])
				else:
					iterator+=1

			nodeFrom = thisNode.leadingEdge[1] #Basically the city index of the current city
			if (len(children) == 0): #If tour is complete
				newBSSF = thisNode.lowerBound + thisNode.matrix[nodeFrom][0]
				if (newBSSF <= BSSF):
					BSSF = newBSSF #Set new BSSF since this is a better (or equal) solution with lower cost
					SolutionNodeSoFar = thisNode

			else:	#Tour is not complete
				for i in range(len(children)):
					thisChild = children[i]
					
					newBound = thisNode.lowerBound #Lower bound of parent
					distTo = thisNode.matrix[nodeFrom][thisChild] #Dist to this child
					newBound = newBound + distTo

					newMatrix = copy.deepcopy(thisNode.matrix)

					for j in range(len(newMatrix[0])): #Set the row of the node from to all infinity
						newMatrix[nodeFrom][j] = float('inf')
					for ki in range(len(newMatrix)): #Set column of the node to to all infinity
						newMatrix[ki][thisChild] = float('inf')
					newMatrix[thisChild][nodeFrom] = float('inf')

					reduced = self.reducedCostMatrix(newMatrix)
					newBound = newBound + reduced[1] #New bound will be the cost of reducing this new matrix
					newMatrix = reduced[0] #.copy() ? FIXME

					if (newBound > BSSF): #Solution is not worth pursuing, so prune the branch
						pruned += 1
						
					else:
						#This else statement is for putting a new child on the queue, and constructing the node and its key on the queue
						newPathSoFar = copy.deepcopy(thisNode.pathSoFar)
						newPathSoFar.append(thisChild)
						depth = len(AdjMatrix)-len(newPathSoFar)
						key = tuple((depth,newBound))

						heapq.heappush(goodQueue, (key, Node(newBound,newMatrix, \
							newPathSoFar,nodeFrom,thisChild)))
						total += 1
						if (len(goodQueue) > theMax):
							theMax = len(goodQueue)

		theTour = None
		tour = []
		tspSolution = None

		#If there is a solution, then construct the tour
		if (SolutionNodeSoFar != None):
			theTour = SolutionNodeSoFar.pathSoFar
			for i in range(len(theTour)):
				tour.append(cities[theTour[i]])
			tspSolution = TSPSolution(tour)
		else:
			print('Solution not found!')
		
		
		end_time = time.time()
		results['cost'] = BSSF if theTour != None else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = tspSolution if theTour != None else None
		results['max'] = theMax
		results['total'] = total
		results['pruned'] = pruned
		return results



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		results = {} #Result of the function
		cities = self._scenario.getCities()
		start_time = time.time()
		count = 0 #How many times the loop has iterated
		
		Best_Greedy = None
		initial_cost = float('inf')
		for i in range(10):
			Greedy_Result = self.greedy(time_allowance)
			if Greedy_Result['cost'] < initial_cost:
				initial_cost = Greedy_Result['cost']
				Best_Greedy = Greedy_Result

		BSSF = initial_cost
		firstRoute = Best_Greedy['soln'] #PUT IN WHILE LOOP TO GET BACK VALID ROUTE

		iterations = len(cities)

		tabuList = [0] * len(cities)
		tabuTener = math.floor(math.sqrt(len(cities))) #3
		numNeighbors = 5

		bestSolution = firstRoute
		currentSolution = firstRoute #Route
		for i in range(iterations):
			count += 1
			for j in range(len(tabuList)):
				if (tabuList[j] != 0):
					tabuList[j] -= 1
			neighbors = []
			for j in range(numNeighbors):
				city1 = random.choice(currentSolution.route)
				city2 = random.choice(currentSolution.route)
				city3 = random.choice(currentSolution.route)
				while (city1._index == city2._index or city2._index == city3._index or city3._index == city1._index):
					city1 = random.choice(currentSolution.route)
					city2 = random.choice(currentSolution.route)
					city3 = random.choice(currentSolution.route)
						
				city1_index = currentSolution.route.index(city1)
				city2_index = currentSolution.route.index(city2)
				city3_index = currentSolution.route.index(city3)

				newSolution = copy.deepcopy(currentSolution.route)
				if (iterations % 3 != 0):
					coinFlip = random.randint(0, 1)

					if (coinFlip == 0):
					#Do a 1 2 3 -> 3 1 2 Swap
						newSolution[city1_index] = city3
						newSolution[city2_index] = city1
						newSolution[city3_index] = city2

					else:
					#Do a 1 2 3 -> 2 3 1 Swap
						newSolution[city1_index] = city2
						newSolution[city2_index] = city3
						newSolution[city3_index] = city1
				else:
					coinFlip = random.randint(0,2)

					if (coinFlip == 0):
						newSolution[city1_index] = city2
						newSolution[city2_index] = city1
					elif(coinFlip == 1):
						newSolution[city2_index] = city3
						newSolution[city3_index] = city2
					else:
						newSolution[city1_index] = city3
						newSolution[city3_index] = city1
					
				# newSolution[city1_index] = city2
				# newSolution[city2_index] = city1

				tempTSPSolution = TSPSolution(newSolution)
				cost = tempTSPSolution.cost

				neighbors.append((tempTSPSolution,cost,city1._index,city2._index,city3._index))
			minCost = float('inf')
			minSolution  = None
			for j in range(len(neighbors)):
				isTabu = False
				if (tabuList[neighbors[j][2]] != 0 or tabuList[neighbors[j][3]] != 0 or tabuList[neighbors[j][4]]):
					isTabu = True
				if (neighbors[j][1] < minCost and (isTabu == False)):
					minCost = neighbors[j][1]
					minSolution = j
				elif (isTabu and neighbors[j][1] < BSSF and neighbors[j][1] < minCost): #If cost is better than BSSF, than break tabu
					minCost = neighbors[j][1]
					minSolution = j
			#Add Tabus to tabuList
			#We are currently making city1 of the swapped solution tabu FIXME

			if (minSolution != None):
				#If neither of the values were tabu before
				if (tabuList[neighbors[minSolution][2]] == 0 and tabuList[neighbors[minSolution][3]] == 0 and \
					tabuList[neighbors[minSolution][4]] == 0):
					tabuList[neighbors[minSolution][2]] = tabuTener #...then make city 1 tabu

				#Set next 'current solution'
				currentSolution = neighbors[minSolution][0]

				#Replace BSSF when necessary
				if (currentSolution.cost < BSSF):
					BSSF = currentSolution.cost
					print('New BSSF: ',BSSF)
					bestSolution = currentSolution

		end_time = time.time()
		
		results['cost'] = bestSolution.cost if Greedy_Result['cost'] != float('inf') else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bestSolution if Greedy_Result['cost'] != float('inf') else None
		return results


