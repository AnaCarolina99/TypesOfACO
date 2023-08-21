from itertools import count
from sqlite3 import Time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from numba import jit, cuda
from typing import *
import random
import math
import time

#Ordem de complexidade da abordagem gulosa O(num ants x num instances2 *iterations)

# Função para calcular a distância entre todas as instâncias
def get_pairwise_distance(matrix: np.ndarray) -> np.ndarray:
    return euclidean_distances(matrix)  # O(n_instances^2 * num_attributes)

# Função para calcular as taxas de visibilidade com base nas distâncias
def get_visibility_rates_by_distances(distances: np.ndarray) -> np.ndarray:
    visibilities = np.zeros(distances.shape)  # O(n_instances^2)
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if i != j:
                if distances[i, j] == 0:
                    visibilities[i, j] = 0
                else:
                    visibilities[i, j] = 1 / distances[i, j]
    return visibilities

# Função para criar a matriz de formigas
def create_colony(num_ants):
    return np.full((num_ants, num_ants), -1)  # O(1)

# Função para criar trilhas de feromônio
def create_pheromone_trails(search_space: np.ndarray, initial_pheromone: float) -> np.ndarray:
    trails = np.full(search_space.shape, initial_pheromone, dtype=np.float64)  # O(n_instances^2 * num_attributes)
    np.fill_diagonal(trails, 0)  # O(n_instances)
    return trails

# Função para calcular o depósito de feromônio com base nas escolhas das formigas
def get_pheromone_deposit(ant_choices: List[Tuple[int, int]], distances: np.ndarray, deposit_factor: float) -> float:
    tour_length = 0
    for path in ant_choices:
        tour_length += distances[path[0], path[1]]  # O(num_ants * n_instances^2 * num_attributes)

    if tour_length == 0:
        return 0  # O(1)

    if math.isinf(tour_length):
        print('deu muito ruim!')  # O(1)

    return deposit_factor / tour_length  # O(1)

# Função para escolher as próximas instâncias com base nas taxas de feromônio
def get_probabilities_paths_ordered_greedy(ant: np.array, phe_trails) -> Tuple[Tuple[int, Any]]:
    available_instances = np.nonzero(ant < 0)[0]  # O(num_ants * n_instances)

    # Calcula as probabilidades com base nas trilhas de feromônio
    smell = phe_trails[available_instances]  # O(num_ants * n_instances)

    sorted_probabilities = smell.argsort()[::-1]  # Ordena em ordem decrescente com base nos valores de feromônio
    return tuple([(int(i), 1) for i in available_instances[sorted_probabilities]])  # O(num_ants * n_instances * log(n_instances))

# Função para encontrar a melhor solução
def get_best_solution(ant_solutions: np.ndarray, X, Y) -> np.array:
    accuracies = np.zeros(ant_solutions.shape[0], dtype=np.float64)  # O(num_ants)
    best_solution = 0
    for i, solution in enumerate(ant_solutions):
        instances_selected = np.nonzero(solution)[0]  # O(n_instances)
        X_train = X[instances_selected, :]
        Y_train = Y[instances_selected]
        classifier_1nn = KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train)  # O(n_instances * num_attributes)
        Y_pred = classifier_1nn.predict(X)  # O(n_instances * num_attributes)
        accuracy = accuracy_score(Y, Y_pred)  # O(n_instances)
        accuracies[i] = accuracy  # O(num_ants)
        if accuracy > accuracies[best_solution]:  # O(num_ants)
            best_solution = i  # O(num_ants)

    return ant_solutions[best_solution]  # O(n_instances)

# Função principal
def run_colony(X, Y, initial_pheromone, evaporarion_rate, Q):
    distances = get_pairwise_distance(X)  # O(n_instances^2 * num_attributes)
    visibility_rates = get_visibility_rates_by_distances(distances)  # O(n_instances^2)
    the_colony = create_colony(X.shape[0])  # O(1)
    for i in range(X.shape[0]):
        the_colony[i, i] = 1  # O(num_ants)

    ant_choices = [[(i, i)] for i in range(the_colony.shape[0])]  # O(num_ants)
    pheromone_trails = create_pheromone_trails(distances, initial_pheromone)  # O(n_instances^2 * num_attributes)
    while -1 in the_colony:  # Loop executa no máximo num_ants * num_ants vezes

        # Cada formiga escolhe sua próxima instância
        for i, ant in enumerate(the_colony):  # Loop executa num_ants vezes

            if -1 in ant:  # O(1)
                last_choice = ant_choices[i][-1]  # O(1)
                ant_pos = last_choice[1]  # O(1)
                choices = get_probabilities_paths_ordered_greedy(ant, pheromone_trails[ant_pos, :])  # O(num_ants * n_instances * log(n_instances))

                for choice in choices:
                    next_instance = choice[0]  # O(1)
                    probability = choice[1]  # O(1)

                    ajk = random.randint(0, 1)  # O(num_ants)
                    final_probability = probability * ajk  # O(num_ants)
                    if final_probability != 0:
                        ant_choices[i].append((ant_pos, next_instance))  # O(1)
                        the_colony[i, next_instance] = 1  # O(1)
                        break
                    else:
                        the_colony[i, next_instance] = 0  # O(1)

        # Formigas depositam os feromônios
        for i in range(the_colony.shape[0]):  # Loop executa num_ants vezes
            ant_deposit = get_pheromone_deposit(ant_choices[i], distances, Q)  # O(num_ants * n_instances^2 * num_attributes)
            for path in ant_choices[i][1:]:  # Nunca deposita em feromônio onde i == j!
                pheromone_trails[path[0], path[1]] += ant_deposit  # O(num_ants * n_instances^2 * num_attributes)

        # Evaporação dos feromônios
        for i in range(pheromone_trails.shape[0]):  # Loop executa n_instances vezes
            for j in range(pheromone_trails.shape[1]):  # Loop executa n_instances vezes
                pheromone_trails[i, j] = (1 - evaporarion_rate) * pheromone_trails[i, j]  # O(n_instances^2 * num_attributes)

    instances_selected = np.nonzero(get_best_solution(the_colony, X, Y))[0]  # O(n_instances)
    return instances_selected

def main():
    start_time = time.time()
    original_df = pd.read_csv("haberman.csv", sep=';')  # O(n_instances * num_attributes)

    dataframe = pd.read_csv("haberman.csv", sep=';')  # O(n_instances * num_attributes)

    classes = dataframe["class"]  # O(n_instances)
    dataframe = dataframe.drop(columns=["class"])  # O(n_instances * num_attributes)
    initial_pheromone = 1  # O(1)
    Q = 1  # O(1)
    evaporation_rate = 0.1  # O(1)
    print('Iniciando busca')
    indices_selected = run_colony(dataframe.to_numpy(), classes.to_numpy(),
                                  initial_pheromone, evaporation_rate, Q)  # O(n_instances^2 * num_attributes)
    print('Fim da busca')
    print(len(indices_selected))  # O(1)

    reduced_dataframe = original_df.iloc[indices_selected]  # O(n_instances * num_attributes)
    reduced_dataframe.to_csv('Home_reduzido.csv', index=False)  # O(n_instances * num_attributes)
    print("Execução finalizada")
    print("--- %s Horas ---" % ((time.time() - start_time) // 3600))  # O(1)
    print("--- %s Minutos ---" % ((time.time() - start_time) // 60))  # O(1)
    print("--- %s Segundos ---" % (time.time() - start_time))  # O(1)

if __name__ == '__main__':
    main()
