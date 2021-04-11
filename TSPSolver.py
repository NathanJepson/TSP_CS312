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
		
	def branchAndBound( self, time_allowance=60.0 ):
		pass



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
		
		Greedy_Result = self.greedy(time_allowance)

		BSSF = Greedy_Result['cost']
		firstRoute = Greedy_Result['soln'] #PUT IN WHILE LOOP TO GET BACK VALID ROUTE

		iterations = len(cities) + 1

		tabuList = [0] * len(cities)
		tabuTener = math.floor(math.sqrt(len(cities)))
		numNeighbors = 3

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
				while (city1._index == city2._index):
					city2 = random.choice(currentSolution.route)
				city1_index = currentSolution.route.index(city1)
				city2_index = currentSolution.route.index(city2)

				newSolution = copy.deepcopy(currentSolution.route) #FIXME
				newSolution[city1_index] = city2
				newSolution[city2_index] = city1

				tempTSPSolution = TSPSolution(newSolution)
				cost = tempTSPSolution.cost

				neighbors.append((tempTSPSolution,cost,city1._index,city2._index))
			minCost = float('inf')
			minSolution  = None
			for j in range(len(neighbors)):
				isTabu = False
				if (tabuList[neighbors[j][2]] != 0 or tabuList[neighbors[j][3]] != 0):
					isTabu = True
				if (neighbors[j][1] < minCost and (isTabu == False)):
					minCost = neighbors[j][1]
					minSolution = j
				elif (isTabu and neighbors[j][1] < BSSF and neighbors[j][i] < minCost): #If cost is better than BSSF, than break tabu
					minCost = neighbors[j][1]
					minSolution = j
			#Add Tabus to tabuList
			#We are currently making city1 of the swapped solution tabu FIXME

			if (minSolution != None):
				#If neither of the values were tabu before
				if (neighbors[minSolution][2] == 0 and neighbors[minSolution][3] == 0  ):
					tabuList[neighbors[minSolution][2]] = tabuTener #...then make city 1 tabu

				#Set next 'current solution'
				currentSolution = neighbors[minSolution][0]

				#Replace BSSF when necessary
				if (currentSolution.cost < BSSF):
					BSSF = currentSolution.cost
					bestSolution = currentSolution

		end_time = time.time()
		
		results['cost'] = bestSolution.cost if Greedy_Result['cost'] != float('inf') else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bestSolution if Greedy_Result['cost'] != float('inf') else None
		return results


