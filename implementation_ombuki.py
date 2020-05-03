#this is an implementation of "Multi-Objective Genetic Algorithms for Vehicle Routing Problems With Time Windows"
#by Ombuki et. al . It uses Solomon's test instances

#TODO list:
# -> Elite retension (currently, the best solutions are still participating in tournament selection)
# -> mutation - inversion should only happen in a single route (at the moment, we're working with the entire chromosome)
# -> crossover - requests should be re-inserted in the best position (we are currentyl inserting in the first feasible position)
# -> routing scheme - Phase 2 from section 3.7 of the paper is yet to be implemented

from __future__ import division
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import deap
from scipy.spatial import distance_matrix
import array
import random
import json
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from copy import deepcopy
from functools import partial
from operator import attrgetter
import multiprocessing
import time
import itertools
import argparse

random.seed(100)
global start_time
global end_time
global dist_mat
global request_data
global capacity

def chromosome2routes_basic(chromosome,verbose=False):
    global start_time
    global end_time
    global dist_mat
    global request_data
    global capacity

    #list of routes
    routes=[]

    #all routes start at the depot (stop/request 0)
    current_route=[0]
    #number of passengers on the current vehicle
    current_passengers=0
    #elapsed time for the current route
    current_time=start_time

    #loop through the requests, in the order specified by the chromosome
    for temp_request in chromosome:
        if verbose : print("adding request ",temp_request)
        #the last served customer on this route
        prev_request=current_route[-1]

        #time required to travel from the previous location
        travel_time = dist_mat[prev_request][temp_request]

        #if the vehicle arrives early, it must wait till the start of the time window i.e. incurr a time penalty
        tw_start = request_data.loc[temp_request]['tw_start']
        waiting_time = max((tw_start-(current_time+travel_time)),0)

        #condition 1:
        #    vehicle capacity must not be exceeded
        passengers=request_data.loc[temp_request]['demand']
        c1 = (current_passengers + passengers ) <= capacity
        if verbose and not(c1) : print("~~~~~adding request would exceed capacity")

        #condition 2:
        #    vehicle must arrive at the customer's location before the time window closes
        c2 = (current_time+travel_time)<= request_data.loc[temp_request]['tw_end']
        if verbose and not(c2) : print("~~~~~vehicle cannot arrive within time window")

        #condition 3:
        #    vehicle must be able to serve the customer and return to the depot within the scheduling period
        service_time = request_data.loc[temp_request]['service_time']
        c3 = (current_time+travel_time+waiting_time+service_time+dist_mat[temp_request][0])<= end_time
        if verbose and not(c3) : print("~~~~~vehicle would not be able to serve and return to the depot")

        #all 3 conditions must be true for the request to be considered feasible to serve
        #if it is feasible to serve this request in the current route, append it
        if (c1 and c2 and c3):
            current_route.append(temp_request)
            current_time+= (travel_time+waiting_time+service_time)
            current_passengers+= passengers

        #otherwise, end the current route here, and start a new route with this request
        else:
            #routes must always end at the depot too
            current_route.append(0)
            routes.append(current_route)

            #start a new route
            if verbose : print("Starting a new route")
            current_route=[0,temp_request]
            current_passengers= passengers
            travel_time=dist_mat[0][temp_request]
            waiting_time = max((tw_start-(start_time + travel_time)),0)
            current_time = start_time + travel_time + waiting_time + service_time

    current_route.append(0)
    routes.append(current_route)
    return routes

def chromosome2routes_advanced(chromosome,verbose=False):
    global start_time
    global end_time
    global dist_mat
    global request_data
    global capacity

    routes = chromosome2routes_basic(chromosome,verbose)

    #TODO : implement this based on Phase 2 of Section 3.7 by Ombuki et. al

    return routes

def routes2chromosome(routes):
    chromosome = list(itertools.chain(*routes))
    chromosome = [gene for gene in chromosome if gene!=0]
    return creator.Individual(chromosome)

def getRouteDistance(route):
    global dist_mat
    distance=0
    for i in range(1,len(route)):
        distance += dist_mat[route[i-1]][route[i]]
    return distance

def getTotalDistance(routes):
    distance=0
    for route in routes:
        distance += getRouteDistance(route)
    return distance

def getChromosomeFitness(chromosome,verbose=False):
    routes=chromosome2routes_advanced(chromosome,verbose)

    #objective 1:
    #    minimize total distance travelled
    obj1=getTotalDistance(routes)

    #objective 2:
    #    minimize number of vehicles required (i.e number of routes)
    obj2=len(routes)

    return obj1,obj2

#based on tools.selTournament
def selCustom(individuals, k):

    #using DEAP's sortNondominated, we get a list of fronts,
    #where each front 'i' dominates front 'i+1'.
    #we use this to create a new attribute for each individual, called "rank"
    #the rank is then used as the fitness value for the tournament selection,
    #as specified by Ombuki et. al
    pareto_fronts = tools.sortNondominated(individuals, k)
    for front_rank in range(1,len(pareto_fronts)+1):
        front=pareto_fronts[front_rank-1]
        for ind in front:
            setattr(ind, 'rank', front_rank)

    #the first rank is the "elite" (Pareto-optimal) set of solutions
    #to which we want to guarantee a spot in the next generation
    #therefore, we extract them before the tournament selection takes place
    elite = pareto_fronts.pop(0)
    individuals_excluding_elite = [i for i in individuals if i not in elite]
    #we update k, the number of individuals to be chosen by the tournament selection
    k-=len(elite)

    #as specified by the paper
    tournsize=4
    r_thresh=0.8

    chosen = []
    for i in range(k):
        aspirants = tools.selRandom(individuals_excluding_elite, tournsize)
        if random.random() < r_thresh :
            chosen_individual=min(aspirants, key=attrgetter("rank"))
        else:
            chosen_individual = tools.selRandom(aspirants, 1)[0]
        chosen.append(chosen_individual)

    #add in the elite solutions
    chosen+=elite
    return chosen

def removeRequest(routes,request):
    for route in routes:
        if request in route:
            #if the route only contains the request we are trying to remove,
            #remove it entirely
            if len(route)<=3:
                routes.remove(route)
            #otherwise, just remove the request from the route
            else:
                route.remove(request)

    return routes

def isFeasible(route):
    global start_time
    global end_time
    global dist_mat
    global request_data
    global capacity

    current_time=start_time
    current_passengers=0

    for i in range(1,len(route)-1):
        temp_request=route[i]
        #the last served customer on this route
        prev_request=route[i-1]

        #time required to travel from the previous location
        travel_time = dist_mat[prev_request][temp_request]
        #if the vehicle arrives early, it must wait till the start of the time window i.e. incurr a time penalty
        tw_start = request_data.loc[temp_request]['tw_start']
        waiting_time = max((tw_start-(current_time+travel_time)),0)
        #time required to serve the current request
        service_time = request_data.loc[temp_request]['service_time']

        #demand of the current request
        passengers=request_data.loc[temp_request]['demand']

        #update the time taken and vehicle capacity
        current_time+= (travel_time+waiting_time+service_time)
        current_passengers+= passengers

        #consider route infeasible if AT LEAST ONE of the following is true:
        #    1. the vehicle capacity has been exceeded
        #    2. the vehicle cannot reach the next request before its time window closes
        c1=current_passengers>capacity
        next_request=route[i+1]
        c2=(current_time + dist_mat[temp_request][next_request]) > request_data.loc[next_request]['tw_end']
        if (c1 or c2) : return False

    return True

def insertRequest(routes,request):

    #to be set to true if at least one route with a feasible insertion point has been found
    feasible_insertion=False

    #try to find the first feasible insertion point
    for route in routes:
        possible_insertion_points=range(1,len(route))
        for ins_index in possible_insertion_points:
            candidate_route=deepcopy(route)
            candidate_route.insert(ins_index,request)
            if isFeasible(candidate_root):
                route=deepcopy(candidate_route)
                return routes

    #if we get to here, no feasible insertion point was found, so we have to create a new route
    additional_route=[0,request,0]
    routes.append(additional_route)
    return routes

def cxCustom(ind1,ind2):
    ind1_routes = chromosome2routes_advanced(ind1)
    ind2_routes = chromosome2routes_advanced(ind2)

    #pick a random route from the other chromosome
    remove_from_ind2 = random.sample(ind1_routes,1)
    #remove the leading and trailing 0 (ie travelling to/from depot)
    remove_from_ind2 = remove_from_ind2[1:-1]
    #repeat for the other chromosome
    remove_from_ind1 = random.sample(ind2_routes,1)
    remove_from_ind1 = remove_from_ind1[1:-1]

    #remove the chosen requests from their original place
    for r in remove_from_ind2 :
        ind2_routes = removeRequest(ind2_routes,r)
    #the removed requests should be re-inserted in a random order
    random.shuffle(remove_from_ind2)
    #re-insert the removed requests
    for r in remove_from_ind2 :
        ind2_routes = insertRequest(ind2_routes,r)

    #repeat for the other chromosome
    for r in remove_from_ind1 :
        ind1_routes = removeRequest(ind1_routes,r)
    random.shuffle(remove_from_ind1)
    for r in remove_from_ind1 :
        ind1_routes = insertRequest(ind1_routes,r)

    return routes2chromosome(ind1_routes), routes2chromosome(ind2_routes)

def mutCustom(individual):

    flip_length = random.choice([2,3])
    flip_start = random.randint(0,(len(individual)-flip_length))
    flip_end = flip_start + (flip_length-1)
    to_flip=individual[flip_start:flip_end+1]
    to_flip.reverse()
    individual=individual[:flip_start]+to_flip+individual[flip_end+1:]
    individual=creator.Individual(individual)

    return individual,

def runGA():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))

    #we specify that the individuals shall be of type "array"
    creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    # Attribute generator
    requests_to_order=request_data.index[1:]
    toolbox.register("indices", random.sample, list(requests_to_order), len(requests_to_order))

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", cxCustom)
    toolbox.register("mutate", mutCustom)
    toolbox.register("select", selCustom)
    toolbox.register("evaluate", getChromosomeFitness)

    pop = toolbox.population(n=500)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean,axis=0)
    #stats.register("std", np.std,axis=0)
    stats.register("min", np.min,axis=0)
    #stats.register("max", np.max,axis=0)

    start = time.time()
    out_pop,out_stats = algorithms.eaSimple(pop,
                                    toolbox,
                                    cxpb=0.8,#crossover probability
                                    mutpb=0.1,#mutation probability
                                    ngen=350,#number of generations
                                    stats=stats,
                                    halloffame=hof)
    end = time.time()
    print("Execution Time : ",end-start," seconds")

    return hof[0]

def visualize(sol):
    plt.figure(figsize=(10,10))
    plt.scatter(request_data['x'][0],request_data['y'][0],c='red')
    plt.scatter(request_data['x'][1:],request_data['y'][1:],c='blue')

    for route in sol:
        previous_x=request_data['x'][route[0]]
        previous_y=request_data['y'][route[0]]
        for point in route[1:]:
            current_x=request_data['x'][point]
            current_y=request_data['y'][point]

            plt.plot([previous_x,current_x], [previous_y,current_y],c='black')

            previous_x=current_x
            previous_y=current_y

    plt.show()

def main(input_filename):
    global start_time
    global end_time
    global dist_mat
    global request_data
    global capacity

    print("reading instance from %s ..."%input_filename)
    input_file = open(input_filename)
    all_lines = input_file.readlines()
    problem_name=all_lines[0]
    max_vehicles = int(all_lines[4].split()[0])
    capacity = int(all_lines[4].split()[1])

    data = np.loadtxt(input_filename,skiprows=8)
    request_data = pd.DataFrame(data,dtype=int,columns=['id','x','y','demand','tw_start','tw_end','service_time'])
    request_data.set_index('id', inplace=True)

    dist_mat = distance_matrix(request_data[['x','y']], request_data[['x','y']])

    # the time window for the depot (request 0) indicates the total scheduling period
    start_time = request_data.loc[0]['tw_start']
    end_time = request_data.loc[0]['tw_end']

    best = runGA()
    best_route=chromosome2routes_basic(best)
    print("\n\nBest Solution:")
    print("Num. Vehivles : ",len(best_route))
    print("Total Distance Travelled : ",getTotalDistance(best_route))
    visualize(best_route)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to the instance file')
    args = parser.parse_args()
    main(args.path)
