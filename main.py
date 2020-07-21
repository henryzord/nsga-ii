# encoding=utf-8

import copy

import numpy as np
from matplotlib import pyplot as plt, cm, markers
from matplotlib.colors import to_hex

__author__ = 'Henry Cagnini'


def a_dominates_b(a, b):
    """
    Checks whether a dominates b (i.e. a is a better solution, in all criteria, than b).

    :param a: first solution
    :type a: numpy.ndarray
    :param b: second solution
    :type b: numpy.ndarray
    :return: -1 if b dominates a, +1 if a dominates b, and 0 if there is no dominance
    """
    a_dominates = any(a > b) and all(a >= b)
    b_dominates = any(b > a) and all(b >= a)

    res = (a_dominates * 1) + (b_dominates * -1)
    return res


def crowding_distance_assignment(_set):
    """
    Worst case scenario for this function: O(m * N * log(N)), where m
    is the number of objectives and N the size of population.

    :type _set: numpy.ndarray
    :param _set: An numpy.ndarray with two dimensions, where the first value in the tuple is from the first objective,
        and so on and so forth.
    :return: A list of crowding distances.
    """

    n_individuals, n_objectives = _set.shape
    crowd_dists = np.zeros(n_individuals)

    for objective in range(n_objectives):
        _set_obj = sorted(_set, key=lambda x: x[objective])
        crowd_dists[[0, -1]] = [np.inf, np.inf]
        for i in range(1, n_individuals-1):
            crowd_dists[i] = crowd_dists[i] + (_set_obj[i+1][objective] - _set_obj[i-1][objective])

    return crowd_dists


def get_fronts(pop: np.ndarray) -> list:
    """
    Given a set of solutions (as a matrix, where each row is an individual and each value in the columns is
    how good that individual is for that criterion), orders solutions based on NSGA-II.

    :param pop: Solutions as a matrix
    :type pop: numpy.ndarray
    :rtype: list
    :return: a list of lists, where each list is a set of solutions. The first list is the first front, and so on
    """

    n_solutions, n_objectives = pop.shape

    dominated = np.zeros(n_solutions, dtype=np.int32)
    dominates = [[] for x in range(n_solutions)]
    fronts = []

    cur_front = []

    for i in range(n_solutions):
        for j in range(i + 1, n_solutions):
            res = a_dominates_b(pop[i], pop[j])
            if res == 1:  # if solution i dominates solution j
                dominated[j] += 1  # signals that j is dominated by one solution
                dominates[i] += [j]  # add j to the list of dominated solutions by i
            elif res == -1:
                dominated[i] += 1  # signals that i is dominated by one solution
                dominates[j] += [i]  # add i to the list of dominated solutions by j

        if dominated[i] == 0:
            cur_front += [i]

    while len(cur_front) != 0:
        some_set = []

        for master in cur_front:
            for slave in dominates[master]:
                dominated[slave] -= 1
                if dominated[slave] == 0:
                    some_set += [slave]

        fronts += [copy.deepcopy(cur_front)]
        cur_front = some_set

    return fronts


def plot_fronts(pop: np.ndarray, fronts: list):
    fig, ax = plt.subplots()

    colors = list(map(to_hex, cm.viridis(np.linspace(0, 1, len(fronts) * 5))[::5]))
    some_markers = markers.MarkerStyle.markers.keys()

    # with sorting and plotting
    for i, (front, color, form) in enumerate(zip(fronts, colors, some_markers)):
        _sorted = np.argsort(pop[front], axis=0)

        ax.scatter(
            (pop[front])[_sorted, 0],
            (pop[front])[_sorted, 1],
            c=color, marker=form, s=45,
            label='Front %d' % (i + 1),
            zorder=1
        )
        ax.plot((pop[front])[_sorted, 0], (pop[front])[_sorted, 1], c=color, zorder=0)

    ax.set_xlabel('Objective #1')
    ax.set_ylabel('Objective #2')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(5)

    # samples population of solutions
    n_individuals = 50
    pop = np.random.random(size=(n_individuals, 2))

    fronts = get_fronts(pop)
    plot_fronts(pop, fronts)


if __name__ == '__main__':
    main()
