import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from typing import List, Tuple
import time
import random

# Calcula a matriz de distâncias euclidianas entre as instâncias
# Complexidade: O(Num_instances² * Num_attributes)
def get_pairwise_distance(matrix: np.ndarray) -> np.ndarray:
    return euclidean_distances(matrix)

# Calcula a matriz de visibilidade baseada nas distâncias
# Complexidade: O(Num_instances²)
def get_visibility_rates_by_distances(distances: np.ndarray) -> np.ndarray:
    visibilities = np.zeros(distances.shape)
    for i in range(distances.shape[0]):  # O(Num_instances)
        for j in range(distances.shape[1]):  # O(Num_instances)
            if i != j:
                visibilities[i, j] = 1 / distances[i, j] if distances[i, j] != 0 else 0
    return visibilities

# Cria uma colônia de formigas (cada linha representa uma formiga)
# Complexidade: O(Num_instances²)
def create_colony(num_ants: int) -> np.ndarray:
    return np.full((num_ants, num_ants), -1)

# Cria trilhas de feromônio com valor inicial
# Complexidade: O(Num_instances²)
def create_pheromone_trails(search_space: np.ndarray, initial_pheromone: float) -> np.ndarray:
    trails = np.full(search_space.shape, initial_pheromone, dtype=np.float64)
    np.fill_diagonal(trails, 0)  # O(Num_instances)
    return trails

# Calcula o feromônio a ser depositado com base nas escolhas da formiga
# Complexidade: O(Num_instances) (pior caso: formiga percorre todas as instâncias)
def get_pheromone_deposit(ant_choices: List[Tuple[int, int]], distances: np.ndarray, deposit_factor: float) -> float:
    tour_length = 0
    for idx, path in enumerate(ant_choices[1:]):  # O(Num_instances)
        tour_length += distances[path[0], path[1]] * (1 - idx / len(ant_choices))
    return deposit_factor / tour_length if tour_length != 0 else 0

# Seleciona a próxima instância baseada em visibilidade e feromônio
# Complexidade: O(Num_instances)
def dynamic_programming_selection(ant: np.array, visibility_rates: np.array, phe_trails: np.array, threshold=0.15) -> Tuple[int, float]:
    available_instances = np.nonzero(ant < 0)[0]  # O(Num_instances)
    if len(available_instances) == 0:
        return -1, 0.0

    values = phe_trails[available_instances] * visibility_rates[available_instances]  # O(Num_instances)
    total_sum = np.sum(values)  # O(Num_instances)
    if total_sum == 0:
        return -1, 0.0

    max_index = np.argmax(values)  # O(Num_instances)
    max_value = values[max_index]

    if (max_value / total_sum) < threshold:
        return -1, 0.0

    return available_instances[max_index], max_value / total_sum

# Avalia a acurácia de cada formiga e retorna a melhor solução
# Complexidade: O(Num_instances² * Num_attributes) (classificação 1-NN)
def get_best_solution(ant_solutions: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.array:
    accuracies = np.zeros(ant_solutions.shape[0], dtype=np.float64)
    best_solution = 0
    for i, solution in enumerate(ant_solutions):  # O(Num_instances)
        instances_selected = np.nonzero(solution)[0]  # O(Num_instances)
        if len(instances_selected) == 0:
            continue
        X_train = X[instances_selected, :]  # O(Num_instances * Num_attributes)
        Y_train = Y[instances_selected]
        classifier_1nn = KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train)  # O(Num_instances² * Num_attributes)
        Y_pred = classifier_1nn.predict(X)  # O(Num_instances * Num_attributes)
        accuracy = accuracy_score(Y, Y_pred)  # O(Num_instances)
        accuracies[i] = accuracy
        if accuracy > accuracies[best_solution]:
            best_solution = i
    return ant_solutions[best_solution]

# Define um número máximo de instâncias a serem selecionadas
# Complexidade: O(1)
def calculate_max_selection(n_instances: int) -> int:
    percentage = random.uniform(0.45, 0.55)
    max_selection = int(round(n_instances * percentage))
    return max_selection

# Executa a lógica principal da colônia de formigas
# Complexidade: O(Num_instances³) (devido às interações entre formigas e atualização de trilhas)
def run_colony(
    X: np.ndarray,
    Y: np.ndarray,
    initial_pheromone: float,
    evaporation_rate: float,
    Q: float,
    threshold: float = 0.15,
    max_selection_per_ant: int = None
) -> np.ndarray:
    distances = get_pairwise_distance(X)  # O(Num_instances² * Num_attributes)
    visibility_rates = get_visibility_rates_by_distances(distances)  # O(Num_instances²)
    n_instances = X.shape[0]

    if max_selection_per_ant is None:
        max_selection_per_ant = calculate_max_selection(n_instances)  # O(1)

    the_colony = create_colony(n_instances)  # O(Num_instances²)

    for i in range(n_instances):  # O(Num_instances)
        the_colony[i, i] = 1

    ant_choices = [[(i, i)] for i in range(n_instances)]  # O(Num_instances)
    pheromone_trails = create_pheromone_trails(distances, initial_pheromone)  # O(Num_instances²)

    active_ants = [True] * n_instances  # O(Num_instances)
    selection_counts = [1] * n_instances  # O(Num_instances)

    while any(active_ants):  # Até todas as formigas terminarem
        for i, ant in enumerate(the_colony):  # O(Num_instances)
            if not active_ants[i]:
                continue

            if selection_counts[i] >= max_selection_per_ant:
                active_ants[i] = False
                the_colony[i, ant < 0] = 0  # O(Num_instances)
                continue

            last_choice = ant_choices[i][-1]
            ant_pos = last_choice[1]

            next_instance, prob = dynamic_programming_selection(
                ant,
                visibility_rates[ant_pos, :],
                pheromone_trails[ant_pos, :],
                threshold=threshold
            )

            if next_instance != -1:
                ant_choices[i].append((ant_pos, next_instance))
                the_colony[i, next_instance] = 1
                selection_counts[i] += 1
            else:
                active_ants[i] = False
                the_colony[i, ant < 0] = 0

        for i in range(n_instances):  # O(Num_instances²)
            ant_deposit = get_pheromone_deposit(ant_choices[i], distances, Q)  # O(Num_instances)
            for path in ant_choices[i][1:]:
                pheromone_trails[path[0], path[1]] += ant_deposit

        pheromone_trails *= (1 - evaporation_rate)  # O(Num_instances²)

    best_ant = get_best_solution(the_colony, X, Y)  # O(Num_instances² * Num_attributes)
    instances_selected = np.nonzero(best_ant)[0]  # O(Num_instances)
    return instances_selected

# Função principal: carrega os dados, executa o algoritmo e salva os resultados
# Complexidade: O(Num_instances³) (dominada por run_colony)
def main():
    start_time = time.time()
    df = pd.read_csv("haberman.csv", sep=';')  # O(Num_instances)
    classes = df["class"].to_numpy()
    X = df.drop(columns=["class"]).to_numpy()

    initial_pheromone = 1.0
    evaporation_rate = 0.1
    Q = 1.0
    threshold = 0.01

    print("Starting search")
    indices_selected = run_colony(X, classes, initial_pheromone, evaporation_rate, Q, threshold)
    print("End Search")

    reduced_df = df.iloc[indices_selected]  # O(Num_instances)
    reduced_df.to_csv("Haberman_reduzido.csv", index=False)  # O(Num_instances)

    print(f"Selected instances: {len(indices_selected)} ({(len(indices_selected)/len(X))*100:.2f}%)")
    print("--- %s Seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()