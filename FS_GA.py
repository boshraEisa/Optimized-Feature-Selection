####################### Feature selection - with and without Genetic algorithms #######################

import random
import numpy as np
from deap import base, creator, algorithms, tools
from FS_RF import *

# Genetic Algorithm constants:

fs = FeatureSelection()
POPULATION_SIZE = 50
P_CROSSOVER = 0.9
P_MUTATION = 0.4
MAX_GENERATIONS = 170

toolbox = base.Toolbox()

# create an operator that randomly returns 0 or 1:
toolbox.register("Binary", random.randint, 0, 1)

# define a single objective, maximum fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.Binary, len(fs.y_test))

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculation
def fitness(individual):
    accuracy = (fs.model.score(fs.X_test, individual))
    return accuracy,


toolbox.register("evaluate", fitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0 / len(fs.y_test))


def main():
    fs = FeatureSelection()
    allOnes = [1] * len(fs)

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, verbose=True)

    print(f'The accuracy score is {round(fs.accuracy(allOnes), 5)} when using all {len(fs.df.columns)} columns')
    print(f"However, using GA, the accuracy score is {round((logbook.select('avg')[170]), 5)} when using only optimal columns ")



if __name__ == "__main__":
    main()
