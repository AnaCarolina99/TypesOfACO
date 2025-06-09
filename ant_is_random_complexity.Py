from itertools import count  # O(1)
from sqlite3 import Time  # O(1)
import pandas as pd  # O(1)
import numpy as np  # O(1)
from sklearn.metrics.pairwise import euclidean_distances  # O(num_instances * num_attributes)
from sklearn.neighbors import KNeighborsClassifier  # O(1)
from sklearn.metrics import accuracy_score  # O(1)
from typing import *  # O(1)
import random  # O(1)
import math  # O(1)
import time  # O(1)

# Ordem de complexidade da abordagem aleatória: O(num_ants × num_instances²)

# Função para calcular a distância entre todas as instâncias
# Complexidade: O(num_instances * num_attributes)
def get_pairwise_distance(matrix: np.ndarray) -> np.ndarray:
    return euclidean_distances(matrix)  # O(num_instances * num_attributes)

# Função para calcular as taxas de visibilidade com base nas distâncias
# Complexidade: O(num_instances)
def get_visibility_rates_by_distances(distances: np.ndarray) -> np.ndarray:

    visibilities = np.where(distances == 0, np.inf, 1 / distances)
    

    np.fill_diagonal(visibilities, 0)
    
    return visibilities


# Função para criar a matriz de formigas
# Complexidade: O(1)
def create_colony(num_ants):
    return np.full((num_ants, num_ants), -1)  # O(1)

# Função para criar trilhas de feromônio
# Complexidade: O(num_instances * num_attributes)
def create_pheromone_trails(search_space: np.ndarray, initial_pheromone: float) -> np.ndarray:
    trails = np.full(search_space.shape, initial_pheromone, dtype=np.float64)  # O(num_instances * num_attributes)
    np.fill_diagonal(trails, 0)  # O(num_instances)
    return trails

# Função para calcular o depósito de feromônio com base nas escolhas das formigas
# Complexidade: O(num_ants * num_instances * num_attributes)
def get_pheromone_deposit(ant_choices: List[Tuple[int, int]], distances: np.ndarray, deposit_factor: float) -> float:
    tour_length = 0
    for path in ant_choices:
        tour_length += distances[path[0], path[1]]  # O(num_ants * num_instances * num_attributes)

    if tour_length == 0:
        return 0  # O(1)

    if math.isinf(tour_length):
        print('deu muito ruim!')  # O(1)

    return deposit_factor / tour_length  # O(1)

# Função para escolher as próximas instâncias com base nas taxas de feromônio
# Complexidade: O(num_ants * num_instances)
def get_probabilities_paths_random(ant: np.array, phe_trails) -> Tuple[Tuple[int, Any]]:
    available_instances = np.nonzero(ant < 0)[0]  # O(num_ants * num_instances)
    
    # Embaralha aleatoriamente as instâncias disponíveis
    np.random.shuffle(available_instances)  # O(num_instances)
    
    # Retorna as instâncias com probabilidades uniformes (1 para todas)
    return tuple([(int(i), 1) for i in available_instances])  # O(num_instances)

# Função para encontrar a melhor solução
# Complexidade: O(num_ants * num_instances * num_attributes)
def get_best_solution(ant_solutions: np.ndarray, X, Y) -> np.array:
    accuracies = np.zeros(ant_solutions.shape[0], dtype=np.float64)  # O(num_ants)
    best_solution = 0
    for i, solution in enumerate(ant_solutions):
        instances_selected = np.nonzero(solution)[0]  # O(num_instances)
        X_train = X[instances_selected, :]
        Y_train = Y[instances_selected]
        classifier_1nn = KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train)  # O(num_instances * num_attributes)
        Y_pred = classifier_1nn.predict(X)  # O(num_instances * num_attributes)
        accuracy = accuracy_score(Y, Y_pred)  # O(num_instances)
        accuracies[i] = accuracy  # O(num_ants)
        if accuracy > accuracies[best_solution]:  # O(num_ants)
            best_solution = i  # O(num_ants)

    return ant_solutions[best_solution]  # O(num_instances)

# Função para executar o algoritmo ACO (Ant Colony Optimization)
# Complexidade: O(num_ants * num_instances * num_attributes)
def run_colony(X, Y, prob_select: float = 0.5) -> np.ndarray:
    num_instances = X.shape[0]

    # Número de formigas aleatório entre 1% e 100% das instâncias
    num_ants = random.randint(max(1, int(num_instances * 0.01)), num_instances)  # O(1)
    print(f"Número de formigas sorteado: {num_ants}")

    colony = np.random.choice([0, 1], size=(num_ants, num_instances), p=[1 - prob_select, prob_select])  # O(num_ants * num_instances)
    instances_selected = np.nonzero(get_best_solution(colony, X, Y))[0]  # O(num_ants * num_instances * num_attributes)
    return instances_selected

# Função principal
# Complexidade dominante: O(num_ants * num_instances * num_attributes)
def main():
    start_time = time.time()

    original_df = pd.read_csv("BrainStroke.csv", sep=';')  # O(num_instances * num_attributes)
    dataframe = pd.read_csv("BrainStroke.csv", sep=';')  # O(num_instances * num_attributes)
    classes = dataframe["stroke"]  # O(num_instances)
    dataframe = dataframe.drop(columns=["stroke"])  # O(num_instances)

    prob_select = 0.5

    print('Iniciando busca com ACO aleatório')
    indices_selected = run_colony(dataframe.to_numpy(), classes.to_numpy(), prob_select)
    print('Busca finalizada')
    print(f"Total de instâncias selecionadas: {len(indices_selected)}")

    reduced_dataframe = original_df.iloc[indices_selected]  # O(num_instances)
    reduced_dataframe.to_csv('BrainStroke_aleatorio.csv', index=False)  # O(num_instances * num_attributes)

    print("Execução concluída")
    print("--- %s horas ---" % ((time.time() - start_time) // 3600))
    print("--- %s minutos ---" % ((time.time() - start_time) // 60))
    print("--- %s segundos ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
