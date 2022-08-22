####################### Feature selection - with and without Partical Swarm Optimization #######################

"""

Programmer: Ali Hussain Khan
Date of Development: 15/10/2020
This code has been developed according to the procedures mentioned in the following research article:
"Khanesar, M. A., Teshnehlab, M., & Shoorehdeli, M. A. (2007, June).
A novel binary particle swarm optimization.
In 2007 Mediterranean conference on control & automation (pp. 1-6). IEEE."

"""

import numpy as np
from Py_FS.wrapper.population_based.algorithm import Algorithm
from Py_FS.wrapper.population_based._utilities import sort_agents
from Py_FS.wrapper.population_based._transfer_functions import get_trans_function
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from FS_RF import *


class PSO(Algorithm):
    def __init__(self,
                 num_agents,
                 max_iter,
                 train_data,
                 train_label,
                 test_data=None,
                 test_label=None,
                 save_conv_graph=False,
                 seed=0,
                 default_mode=False,
                 verbose=True):

        super().__init__(num_agents=num_agents,
                         max_iter=max_iter,
                         train_data=train_data,
                         train_label=train_label,
                         test_data=test_data,
                         test_label=test_label,
                         save_conv_graph=save_conv_graph,
                         seed=seed,
                         default_mode=default_mode,
                         verbose=verbose)

        self.algo_name = 'PSO'
        self.agent_name = 'Particle'

    def user_input(self):
        # first set the default values for the attributes
        self.default_vals["trans_function"] = 's'

        # accept the parameters as user inputs (if default_mode not set)
        if self.default_mode:
            self.set_default()
        else:
            self.algo_params['trans_function'] = input(
                f"Shape of Transfer Function [s/v/u] (default={self.default_vals['trans_function']}):") or \
                                                 self.default_vals["trans_function"]
        self.trans_function = get_trans_function(self.algo_params['trans_function'])


    def initialize(self):
        super().initialize()
        self.global_best_particle = [0 for i in range(self.num_features)]
        self.global_best_fitness = float("-inf")
        self.local_best_particle = [[0 for i in range(self.num_features)] for j in range(self.num_agents)]
        self.local_best_fitness = [float("-inf") for i in range(self.num_agents)]
        self.velocity = [[0.0 for i in range(self.num_features)] for j in range(self.num_agents)]

        self.weight = 1.0

    def next(self):
        self.print('\n================================================================================')
        self.print('                          Iteration - {}'.format(self.cur_iter + 1))
        self.print('================================================================================\n')

        # update weight
        self.weight = 1.0 - (self.cur_iter / self.max_iter)

        # update the velocity
        for i in range(self.num_agents):
            for j in range(self.num_features):
                self.velocity[i][j] = (self.weight * self.velocity[i][j])
                r1, r2 = np.random.random(2)
                self.velocity[i][j] = self.velocity[i][j] + (
                        r1 * (self.local_best_particle[i][j] - self.population[i][j]))
                self.velocity[i][j] = self.velocity[i][j] + (
                        r2 * (self.global_best_particle[j] - self.population[i][j]))

        # updating position of particles
        for i in range(self.num_agents):
            for j in range(self.num_features):
                trans_value = self.trans_function(self.velocity[i][j])
                if np.random.random() < trans_value:
                    self.population[i][j] = 1
                else:
                    self.population[i][j] = 0

        # updating fitness of particles
        self.fitness = self.obj_function(self.population, self.training_data)
        self.population, self.fitness = sort_agents(agents=self.population, fitness=self.fitness)

        # updating the global best and local best particles
        for i in range(self.num_agents):
            if self.fitness[i] > self.local_best_fitness[i]:
                self.local_best_fitness[i] = self.fitness[i]
                self.local_best_particle[i] = self.population[i][:]

            if self.fitness[i] > self.global_best_fitness:
                self.global_best_fitness = self.fitness[i]
                self.global_best_particle = self.population[i][:]

        self.cur_iter += 1


def main():
    fs = FeatureSelection()
    allOnes = [1] * len(fs)
    #
    df = pd.read_csv('breast-cancer.csv')
    #
    # # change the diagnosis column from M and B to 0 and 1
    diagnosis = {'M': 1, 'B': 0}
    #
    df.diagnosis = [diagnosis[item] for item in df.diagnosis]
    # delete the id columns since it does not provide any value
    del df['id']

    y = df.iloc[:, 0]  # Dependent variable, the diagnosis, is the first one
    X = df.iloc[:, 1:31]  # Independent variables (first and second variable: age and interest)

    algo = PSO(num_agents=30, max_iter=20, train_data=X, train_label=y, default_mode=True)
    solution = algo.run()

    print(f'The accuracy score is {round(fs.accuracy(allOnes), 5)} when using all {len(df.columns)} columns')
    print(f"However, using PSO, the accuracy score is {round(solution.global_best_fitness, 5)} when using only "
          f"optimal columns ")


if __name__ == '__main__':
    main()
