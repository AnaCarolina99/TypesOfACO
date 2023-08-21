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

#Ordem de complexidade da abordagem dinâmica O(num instances^2 x num attributes)

# Função para calcular a distância entre todas as instâncias
# Complexidade: O(num_instances^2 * num_attributes)
def get_pairwise_distance(matrix: np.ndarray) -> np.ndarray:
    return euclidean_distances(matrix)  # O(n_instances^2 * n_attributes)

# Função para calcular as taxas de visibilidade com base nas distâncias
# Complexidade: O(num_instances * num_attributes)
def get_visibility_rates_by_distances(distances: np.ndarray) -> np.ndarray:
    return 1 / np.maximum(np.sum(distances, axis=1), 1e-6)  # O(n_instances * n_attributes)

# Função para calcular o depósito de feromônio com base em um subconjunto de instâncias
# Complexidade: O(num_instances^2 * num_attributes)
def get_pheromone_deposit(instance_subset: np.ndarray, distances: np.ndarray, deposit_factor: float) -> float:
    tour_length = np.sum(distances[instance_subset[:, None], instance_subset])  # O(n_instances^2 * n_attributes)
    if tour_length == 0:  # O(1)
        return 0  # O(1)

    return deposit_factor / tour_length  # O(1)

# Função para seleção usando programação dinâmica
# Complexidade: O(num_instances^2 * num_attributes)
def dynamic_programming_selection(X, Y, distances, visibility_rates, Q):
    num_instances = X.shape[0]
    dp_table = np.zeros((num_instances + 1, num_instances + 1))
    dp_choices = np.zeros((num_instances + 1, num_instances + 1), dtype=int)

    for i in range(1, num_instances + 1):  # Loop O(n_instances)
        for j in range(1, num_instances + 1):  # Loop O(n_instances)
            if i == j:
                continue

            exclude_instance = dp_table[i, j - 1]  # O(1)
            include_instance = dp_table[i - 1, j - 1] + visibility_rates[j - 1] * distances[i - 1, j - 1]  # O(1)
            dp_table[i, j] = max(exclude_instance, include_instance)  # O(1)

            if include_instance > exclude_instance:  # O(1)
                dp_choices[i, j] = 1  # O(1)

    selected_instances = []  # O(1)
    i, j = num_instances, num_instances  # O(1)
    while i > 0 and j > 0:  # Loop O(n_instances)
        if dp_choices[i, j] == 1:  # O(1)
            selected_instances.append(j - 1)  # O(1)
            i -= 1  # O(1)
            j -= 1  # O(1)
        else:
            j -= 1  # O(1)

    selected_instances.reverse()  # O(n_instances)
    return np.array(selected_instances)  # O(n_instances)

# Função principal
def main():
    start_time = time.time()
    
    # Carregar o conjunto de dados
    original_df = pd.read_csv("dna.csv", sep=';')  # O(n_instances * num_attributes)

    # Pré-processamento dos dados
    dataframe = pd.read_csv("dna.csv", sep=';')  # O(n_instances * num_attributes)
    classes = dataframe["class"] # O(n_instances)
    dataframe = dataframe.drop(columns=["class"]) # O(n_instances * num_attributes)

    # Cálculos de distâncias e taxas de visibilidade
    distances = get_pairwise_distance(dataframe.to_numpy())  # O(n_instances^2 * n_attributes)
    visibility_rates = get_visibility_rates_by_distances(distances)  # O(n_instances * n_attributes)

    initial_pheromone = 1   # O(1)
    Q = 1  # O(1)

    print('Starting search')
    indices_selected = dynamic_programming_selection(dataframe.to_numpy(), classes.to_numpy(), distances,
                                                     visibility_rates, Q)  # O(n_instances^2 * n_attributes)
    print('End Search')
    print(len(indices_selected))

    reduced_dataframe = original_df.iloc[indices_selected]
    reduced_dataframe.to_csv('Splice_Dynamic.csv', sep=',', index=False)  

    print("Execution finished")
    print("--- %s Hours ---" % ((time.time() - start_time) // 3600))
    print("--- %s Minutes ---" % ((time.time() - start_time) // 60))
    print("--- %s Seconds ---" % (time.time() - start_time))

# Executa a função principal se este script for executado diretamente
if __name__ == '__main__':
    main()
