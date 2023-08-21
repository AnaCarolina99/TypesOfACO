from itertools import count  # O(1)
from sqlite3 import Time  # O(1)
import pandas as pd  # O(1)
import numpy as np  # O(1)
from sklearn.metrics.pairwise import euclidean_distances  # O(num_instances^2 * num_attributes)
from sklearn.neighbors import KNeighborsClassifier  # O(1)
from sklearn.metrics import accuracy_score  # O(1)
from numba import jit, cuda  # O(1)

from typing import *  # O(1)
import random  # O(1)
import math  # O(1)
import time  # O(1)

# Função para calcular a distância euclidiana entre todas as instâncias.
# Complexidade: O(num_instances^2 * num_attributes)
def get_pairwise_distance(matrix: np.ndarray) -> np.ndarray:
    return euclidean_distances(matrix)  # O(num_instances^2 * num_attributes)

# Função para calcular as taxas de visibilidade com base nas distâncias.
# Complexidade: O(num_instances^2)
def get_visibility_rates_by_distances(distances: np.ndarray) -> np.ndarray:
    visibilities = np.zeros(distances.shape)  # O(num_instances^2)
    for i in range(distances.shape[0]):  # Loop O(num_instances)
        for j in range(distances.shape[1]):  # Loop O(num_instances)
            if i != j:  # O(1)
                if distances[i, j] == 0:  # O(1)
                    visibilities[i, j] = 0  # O(1)
                else:
                    visibilities[i, j] = 1 / distances[i, j]  # O(1)
    return visibilities  # O(1)

# Função para criar uma matriz representando uma colônia de formigas.
# Complexidade: O(num_ants^2)
def create_colony(num_ants):  # O(1)
    return np.full((num_ants, num_ants), -1)  # O(num_ants^2)

# Função para criar matrizes de trilhas de feromônio inicializadas com um valor específico.
# Complexidade: O(num_instances^2)
def create_pheromone_trails(search_space: np.ndarray, initial_pheromone: float) -> np.ndarray:
    trails = np.full(search_space.shape, initial_pheromone, dtype=np.float64)  # O(num_instances^2)
    np.fill_diagonal(trails, 0)  # O(num_instances)
    return trails  # O(1)

# Função para calcular o depósito de feromônio com base em um subconjunto de instâncias.
# Complexidade: O(num_instances)
def get_pheromone_deposit(ant_choices: List[Tuple[int, int]], distances: np.ndarray, deposit_factor: float) -> float:
    tour_length = 0  # O(1)
    for path in ant_choices:  # Loop O(num_ants)
        tour_length += distances[path[0], path[1]]  # O(num_ants * num_instances * num_attributes)
    if tour_length == 0:  # O(1)
        return 0  # O(1)
    return deposit_factor / tour_length  # O(1)

# Função para calcular probabilidades de seleção de caminhos ordenados por cheiro (feromônio).
# Complexidade: O(num_instances * log(num_instances))
def get_probabilities_paths_ordered(ant: np.array, visibility_rates: np.array, phe_trails) -> Tuple[Tuple[int, Any]]:
    available_instances = np.nonzero(ant < 0)[0]  # O(num_instances)
    smell = np.sum(phe_trails[available_instances] * visibility_rates[available_instances])  # O(num_instances * log(num_instances))

    probabilities = np.zeros((len(available_instances), 2))  # O(num_instances)
    for i, available_instance in enumerate(available_instances):  # Loop O(num_instances)
        probabilities[i, 0] = available_instance  # O(1)
        path_smell = phe_trails[available_instance] * visibility_rates[available_instance]  # O(1)
        if path_smell == 0:  # O(1)
            probabilities[i, 1] = 0  # O(1)
        else:
            probabilities[i, 1] = path_smell / smell  # O(1)

    sorted_probabilities = probabilities[probabilities[:, 1].argsort()][::-1]  # O(num_instances * log(num_instances))
    return tuple([(int(i[0]), i[1]) for i in sorted_probabilities])  # O(num_instances * log(num_instances))

# Função para encontrar a melhor solução entre as formigas.
# Complexidade: O(num_ants * num_instances * num_attributes)
def get_best_solution(ant_solutions: np.ndarray, X, Y) -> np.array:
    accuracies = np.zeros(ant_solutions.shape[0], dtype=np.float64)  # O(num_ants)
    best_solution = 0  # O(1)
    for i, solution in enumerate(ant_solutions):  # Loop O(num_ants)
        instances_selected = np.nonzero(solution)[0]  # O(num_instances)
        X_train = X[instances_selected, :]  # O(num_instances * num_attributes)
        Y_train = Y[instances_selected]  # O(num_instances)
        classifier_1nn = KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train)  # O(num_instances * num_attributes)
        Y_pred = classifier_1nn.predict(X)  # O(num_instances * num_attributes)
        accuracy = accuracy_score(Y, Y_pred)  # O(num_instances)
        accuracies[i] = accuracy  # O(1)
        if accuracy > accuracies[best_solution]:  # O(num_ants)
            best_solution = i  # O(1)
    return ant_solutions[best_solution]  # O(1)

# Função principal
def main():
    start_time = time.time()  # O(1)
    original_df = pd.read_csv("heloc_dataset_v1.csv", sep=';')  # O(1)
    dataframe = pd.read_csv("heloc_dataset_v1.csv", sep=';')  # O(1)
    classes = dataframe["PercentTradesWBalance"]  # O(1)
    dataframe = dataframe.drop(columns=["PercentTradesWBalance"])  # O(1)
    initial_pheromone = 1  # O(1)
    Q = 1  # O(1)
    evaporation_rate = 0.1  # O(1)
    print('Iniciando a busca')  # O(1)
    indices_selected = run_colony(dataframe.to_numpy(), classes.to_numpy(),
                                  initial_pheromone, evaporation_rate, Q)  # O(num_ants * num_instances^2 * log(num_instances) + num_ants * num_instances * num_attributes)
    print('Fim da busca')  # O(1)
    print(len(indices_selected))  # O(1)
    reduced_dataframe = original_df.iloc[indices_selected]  # O(num_ants * num_instances * num_attributes)
    reduced_dataframe.to_csv('Home_reduzido.csv', index=False)  # O(num_ants * num_instances * num_attributes)
    print("Execução finalizada")  # O(1)
    print("--- %s Horas ---" % ((time.time() - start_time)//3600))  # O(1)
    print("--- %s Minutos ---" % ((time.time() - start_time)//60))  # O(1)
    print("--- %s Segundos ---" % (time.time() - start_time))  # O(1)

# Executa a função principal se este script for executado diretamente
if __name__ == '__main__':
    main()
