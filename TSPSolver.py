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

		# AdjMatrix = [[float('inf') for j in range(len(cities))] for i in range(len(cities))] #Initialize adjacency matrix

		# for i in range(len(cities)): #Set up the distances of the adjacenty matrix
		# 	for j in range(len(cities)):
		# 		if i != j:			
		# 			dist = cities[i].costTo(cities[j])
		# 			AdjMatrix[i][j] = dist

		# #AdjMatrix = [[float('inf'),7,3,12],[3,float('inf'),6,14],[5,8,float('inf'),6],[9,3,5,float('inf')]]
		# print(AdjMatrix)

		#reducedCost = self.reducedCostMatrix(AdjMatrix) #Function to get reduced cost matrix
		#AdjMatrix = reducedCost[0] #The first result of that function gives the matrix

		#bound = reducedCost[1] #The second result of the function gives the reduction cost

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
		pass