# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from typing import List, Tuple
import time
import random


def get_pairwise_distance(matrix: np.ndarray) -> np.ndarray:
    return euclidean_distances(matrix)


def get_visibility_rates_by_distances(distances: np.ndarray) -> np.ndarray:
    visibilities = np.zeros(distances.shape)
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if i != j:
                visibilities[i, j] = 1 / distances[i, j] if distances[i, j] != 0 else 0
    return visibilities


def create_colony(num_ants: int) -> np.ndarray:
    return np.full((num_ants, num_ants), -1)


def create_pheromone_trails(search_space: np.ndarray, initial_pheromone: float) -> np.ndarray:
    trails = np.full(search_space.shape, initial_pheromone, dtype=np.float64)
    np.fill_diagonal(trails, 0)
    return trails


def get_pheromone_deposit(ant_choices: List[Tuple[int, int]], distances: np.ndarray, deposit_factor: float) -> float:
    tour_length = 0
    for idx, path in enumerate(ant_choices[1:]):
        tour_length += distances[path[0], path[1]] * (1 - idx / len(ant_choices))
    return deposit_factor / tour_length if tour_length != 0 else 0


def dynamic_programming_selection(ant: np.array, visibility_rates: np.array, phe_trails: np.array) -> int:
    available_instances = np.nonzero(ant < 0)[0]
    if len(available_instances) == 0:
        return -1

    values = phe_trails[available_instances] * visibility_rates[available_instances]
    total_sum = np.sum(values)
    if total_sum == 0:
        return -1

    probabilities = values / total_sum
    return np.random.choice(available_instances, p=probabilities)


def get_best_solution(ant_solutions: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.array:
    accuracies = np.zeros(ant_solutions.shape[0], dtype=np.float64)
    best_solution = 0
    for i, solution in enumerate(ant_solutions):
        instances_selected = np.nonzero(solution)[0]
        if len(instances_selected) == 0:
            continue
        X_train = X[instances_selected, :]
        Y_train = Y[instances_selected]
        classifier_1nn = KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train)
        Y_pred = classifier_1nn.predict(X)
        accuracy = accuracy_score(Y, Y_pred)
        accuracies[i] = accuracy
        if accuracy > accuracies[best_solution]:
            best_solution = i
    return ant_solutions[best_solution]


def calculate_max_selection(n_instances: int) -> int:
    percentage = random.uniform(0.45, 0.55)
    return int(round(n_instances * percentage))


def run_colony(
    X: np.ndarray,
    Y: np.ndarray,
    initial_pheromone: float,
    evaporation_rate: float,
    Q: float,
    max_selection_per_ant: int = None,
    total_instances_to_select: int = None
) -> np.ndarray:
    distances = get_pairwise_distance(X)
    visibility_rates = get_visibility_rates_by_distances(distances)
    n_instances = X.shape[0]

    if max_selection_per_ant is None:
        max_selection_per_ant = calculate_max_selection(n_instances)

    the_colony = create_colony(n_instances)
    for i in range(n_instances):
        the_colony[i, i] = 1

    ant_choices = [[(i, i)] for i in range(n_instances)]
    pheromone_trails = create_pheromone_trails(distances, initial_pheromone)

    active_ants = [True] * n_instances
    selection_counts = [1] * n_instances

    while any(active_ants):
        for i, ant in enumerate(the_colony):
            if not active_ants[i]:
                continue

            if selection_counts[i] >= max_selection_per_ant:
                active_ants[i] = False
                the_colony[i, ant < 0] = 0
                continue

            last_choice = ant_choices[i][-1]
            ant_pos = last_choice[1]

            next_instance = dynamic_programming_selection(
                ant,
                visibility_rates[ant_pos, :],
                pheromone_trails[ant_pos, :]
            )

            if next_instance != -1:
                ant_choices[i].append((ant_pos, next_instance))
                the_colony[i, next_instance] = 1
                selection_counts[i] += 1
            else:
                active_ants[i] = False
                the_colony[i, ant < 0] = 0

        for i in range(n_instances):
            ant_deposit = get_pheromone_deposit(ant_choices[i], distances, Q)
            for path in ant_choices[i][1:]:
                pheromone_trails[path[0], path[1]] += ant_deposit

        pheromone_trails *= (1 - evaporation_rate)

    best_ant = get_best_solution(the_colony, X, Y)
    instances_selected = np.nonzero(best_ant)[0]

    if total_instances_to_select is not None:
        if total_instances_to_select > len(instances_selected):
            total_instances_to_select = len(instances_selected)
        final_selected = np.random.choice(instances_selected, size=total_instances_to_select, replace=False)
    else:
        final_percentage = random.uniform(0.95, 0.99)
        final_count = int(round(len(instances_selected) * final_percentage))
        final_selected = np.random.choice(instances_selected, size=final_count, replace=False)

    return final_selected


def main():
    start_time = time.time()
    df = pd.read_csv("BrainStroke.csv", sep=';')
    classes = df["stroke"].to_numpy()
    X = df.drop(columns=["stroke"]).to_numpy()

    initial_pheromone = 1.0
    evaporation_rate = 0.1
    Q = 1.0

    total_instances_to_select = 2851 # ? Defina aqui o total desejado

    print("Starting search")
    indices_selected = run_colony(
        X, classes,
        initial_pheromone,
        evaporation_rate,
        Q,
        total_instances_to_select=total_instances_to_select
    )
    print("End Search")

    reduced_df = df.iloc[indices_selected]
    reduced_df.to_csv("BrainStroke_Dinamico.csv", index=False)

    print(f"Selected instances: {len(indices_selected)} ({(len(indices_selected)/len(X))*100:.2f}%)")
    print("--- %s Seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
