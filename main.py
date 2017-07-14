# encoding=utf-8

import numpy as np
import operator as op
from matplotlib import pyplot as plt

__author__ = 'Henry Cagnini'


def a_dominates_b(a, b):
    """

    :param a: first solution
    :param b: second solution
    :return: -1 if b dominates a, +1 if the opposite, and 0 if there is no dominance
    """
    a_dominates = any(a > b) and all(a >= b)
    b_dominates = any(b > a) and all(b >= a)

    res = (a_dominates * 1) + (b_dominates * -1)
    return res


def get_fronts(pop):
    n_individuals, n_objectives = pop.shape

    added = np.zeros(n_individuals)
    dominated = np.zeros(n_individuals)
    dominates = [[] for x in xrange(n_individuals)]
    fronts = []

    cur_front = 0

    for i in xrange(n_individuals):
        for j in xrange(i+1, n_individuals):
            res = a_dominates_b(pop[i], pop[j])
            if res == 1:
                dominated[j] += 1
                dominates[i] += [j]
            elif res == -1:
                dominated[i] += 1
                dominates[j] += [i]

    while sum(added) < n_individuals:
        _where = np.flatnonzero(dominated == 0)

        if len(_where) == 0:
            break

        added[_where] = 1
        fronts += [pop[_where]]
        dominated[_where] = -1

        _chain = set(reduce(op.add, map(lambda x: dominates[x], _where)))

        for k in _chain:
            dominated[k] -= 1

        cur_front += 1

    return fronts
    

def crowding_distance_assignment(_set):
    """
    Worst case scenario for this function: O(m * N * log(N)), where m
    is the number of objectives and N the size of population.

    :type set: numpy.ndarray
    :param _set: An numpy.ndarray with two dimensions, where the first value in the tuple is from the first objective,
        and so on and so forth.
    :return: A list of crowding distances.
    """

    n_individuals, n_objectives = _set.shape
    crowd_dists = np.zeros(n_individuals)

    for objective in xrange(n_objectives):
        _set_obj = sorted(_set, key=lambda x: x[objective])
        crowd_dists[[0, -1]] = [np.inf, np.inf]
        for i in xrange(1, n_individuals-1):
            crowd_dists[i] = crowd_dists[i] + (_set_obj[i+1][objective] - _set_obj[i-1][objective])

    return crowd_dists


def a_greater_b(rank_a, crowd_dist_a, rank_b, crowd_dist_b):
    return (rank_a < rank_b) or ((rank_a == rank_b) and (crowd_dist_a > crowd_dist_b))


def main():
    # np.random.seed(3)

    x_axis = range(10)

    n_objectives = 2
    n_individuals = len(x_axis) * 2

    pop = np.random.random(size=(n_individuals, n_objectives))

    fronts = get_fronts(pop)

    for j in xrange(pop.shape[0]):
        plt.text(
            x=pop[j, 0],
            y=pop[j, 1],
            s=str(j)
        )

    plt.scatter(pop[:, 0], pop[:, 1], c='red', label='non-front population')

    for i, front in enumerate(fronts):
        color = "#%06x" % np.random.randint(0, 0xFFFFFF)

        plt.scatter(
            fronts[i][:, 0],  # X
            fronts[i][:, 1],  # Y
            c=color,
            label='%d-th front' % i
        )

        _sorted = np.asarray(sorted(fronts[i], key=lambda x: x[0]))

        plt.plot(
            _sorted[:, 0],  # X
            _sorted[:, 1],  # Y
            c=color
        )

        f1_dists = crowding_distance_assignment(fronts[i])
        print '%d-th front crowding distances:', f1_dists

    plt.title('Pareto Front')
    plt.xlabel('objective x')
    plt.ylabel('objective y')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
