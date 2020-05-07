the data used is from Solomon's test instances for the VRPTW, obtained from http://neo.lcc.uma.es/vrp/

-------Summary Ombuki et. al-------
static
Multi-Objective: Minimize # of vehicles and travel distance

Chromosome Representation
-> a chromosome is a 1D array which only represents the order in which requests are visited (not the routes)
-> chromosomes are converted to a set of routes on the network as follows:
  -> Phase 1: keep inserting requests into the same route until the time windows or vehicle capacity are violated. The request which causes this to happen should be the start of a new routes
  -> Phase 2: the last customer of each route is removed and placed at the very start of the next route. If this is advantageous, keep it this way, else revert (not implemented yet)

Fitness Evaluation
1. convert chromosome to routes
2. count the number of routes in the chromosome (i.e. number of vehicles required)
3. evaluate the total distance covered by all the routes

Selection Procedure
-> selection is carried out based on the Pareto rank of each individual (i.e. we first carry out non-dominated sorting of the population)
-> a selection tournament of size 4 is used to select the fittest individuals, picking the individual with the lowest rank each time
-> however, there is a 20% chance that the tournament winner is chosen at random instead
-> because of this, we first exclude the elite solutions (rank 1) from the tournament selection process, and put them immediately into the offspring population.
   The tournament selection is then used to fill out the rest of the offspring.

Crossover
-> Best-Cost Route Crossover (BCRC) - see Figure 5
-> Crossover happens between two individuals in the offspring population
-> For each pair of individuals A and B:
      1. Pick a route in individual A
      2. Remove the requests in that route from individual B
      3. Try to re-insert them (into individual B) in a randomized order, each time picking the best insertion location (greedy)
      4. Switch A and B and repeat

Mutation
-> Mutation is carried out by switching around only 2 or 3 requests in the entire chromosome
-> The paper specifies that mutation should only modify requests which are in the same route. However, when this was implemented, the solutions generated were actually worse.
   NB. This change is in the feature_SingleRouteInversionMutation branch, but hasn't been merged to master for this reason.

Other Notes
-> the paper implies that the elite solutions are not modified at all, meaning that they are probably excluded from the Crossover/Mutation part.
   We would need to re-write a modified version of eaSimple to implement this kind of elite retention (should be quite easy to do)

----------------------------------------
