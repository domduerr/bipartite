import glob
import math
import multiprocessing as mp
import random
import sqlite3
import time
from subprocess import TimeoutExpired

import numpy as np
from bs4 import BeautifulSoup
from satispy import Variable
from satispy.solver import Minisat

# Initialize some public objects
solver = Minisat(timeout=60*60)
con = sqlite3.connect("data.db", check_same_thread=False)
cur = con.cursor()
mutex = mp.Lock()


def solve_sat(g, k):
    """Check whether the graph g has a set of vertices of size k which can be
    deleted to make the graph bipartite"""

    # Generate Variables
    exp = Variable('e')
    v = {x: [Variable(str(x)+'_1'), Variable(str(x)+'_2'),
             Variable(str(x)+'_3')]
         for x in range(len(g))}

    # Clauses that guarantee that the split into A and B is valid
    for x in g:
        exp &= v[x][0] | v[x][1] | v[x][2]
    for x in g:
        for y in g[x]:
            if x < y:
                exp &= -v[x][0] | -v[y][0]
                exp &= -v[x][1] | -v[y][1]
    if k == 0:
        # If k is zero, there can't be any vertices in D
        for x in range(len(g)):
            exp &= -v[x][2]
    else:
        # Limit the size of D to k, using the clauses of Sinz
        # (https://www.carstensinz.de/papers/CP-2005.pdf)
        r = {i: {j: Variable('R_' + str(i) + "_" + str(j))
                 for j in range(k)}
             for i in range(len(g)-1)}
        exp &= -v[0][2] | r[0][0]
        for j in range(1, k):
            exp &= -r[0][j]
        for i in range(1, len(g)-1):
            exp &= -v[i][2] | r[i][0]
            exp &= -r[i-1][0] | r[i][0]
            for j in range(1, k):
                exp &= -v[i][2] | -r[i-1][j-1] | r[i][j]
                exp &= -r[i-1][j] | r[i][j]
            exp &= -v[i][2] | -r[i-1][k-1]
        exp &= -v[len(g)-1][2] | -r[len(g)-2][k-1]

    # Run the SAT solver
    solution = solver.solve(exp)

    # Return whether a solution is found
    return solution.success


def sat(g):
    """Binary search to check for solutions of the problem with the SAT
    solver."""
    upper = 1
    lower = 0
    while not solve_sat(g, upper):
        upper *= 2
    while not lower == upper:
        if lower + 1 == upper:
            if solve_sat(g, lower):
                return lower
            else:
                return upper
        k = int((upper + lower)/2)
        test = solve_sat(g, k)
        if test:
            upper = k
        else:
            lower = k
    return lower


def solve_greedy(g, selected=None):
    """Greedy algorithm for computing an inclusion-maximal bipartite
    subgraph. Basically a breath-first-search that assigns colors to the
    vertices in alternating order."""
    if selected is None:
        selected = np.zeros(len(g), dtype=int)
    nonselected = np.where(np.zeros(len(g), dtype=int) == 0)[0]
    np.random.shuffle(nonselected)
    for x in nonselected:
        if selected[x] != 0:
            continue
        possible_color = {1, 2}.difference(
            {selected[y] for y in g[x]}.difference())
        if len(possible_color) == 0:
            continue
        todo = [x]
        selected[x] = random.choice(tuple(possible_color))
        while len(todo) > 0:
            u = todo.pop()
            if selected[u] in {selected[v] for v in g[u]}:
                selected[u] = 0
            else:
                for v in g[u]:
                    if selected[v] == 0:
                        todo.append(v)
                        selected[v] = 2 if selected[u] == 1 else 1
    return selected


def greedy(g):
    """Returns the size of the set D for the greedy algorithm."""
    return len(g) - np.count_nonzero(solve_greedy(g))


def cooling1(t, t0, n):
    """Linear cooling function for the simulated annealing algorithm."""
    return t0 * ((n-t) / n)


def cooling2(t, t0, n):
    """Quadratic cooling function for the simulated annealing algorithm."""
    return t0 * (((n-t) / n)**2)


def cooling3(t, t0, n):
    """Exponential cooling function for the simulated annealing algorithm."""
    return t0 * (1 / (1 + math.exp((2 * math.log(t0))/n*(t-n/2))))


def cooling4(t, t0, n):
    """Hill-Climbing cooling function for the simulated annealing algorithm."""
    return 0


def solve_simulated_annealing(g, max_iter, init_temp, flip_prob, cooling):
    """Simulated annealing algorithm for computing an inclusion-maximal
    bipartite subgraph."""
    # Save the best achieved result
    m = len(g)

    # Compute a starting solution
    current = solve_greedy(g)

    # Iterations
    for t in range(max_iter):
        # Pick a neighbor
        a = np.nonzero(current == 0)[0]
        if len(a) == 0:
            return (current, 0)
        u = np.random.choice(a)
        candidate = np.copy(current)
        for v in g[u]:
            candidate[v] = 0
        candidate[u] = random.randint(0, 1)+1
        candidate = solve_greedy(g, candidate)

        # Do we accept the neighbor?
        fx = len(g) - np.count_nonzero(current)
        fy = len(g) - np.count_nonzero(candidate)
        m = min(m, fy)
        diff = fx - fy
        c = cooling(t, init_temp, max_iter)
        if c == 0:
            if diff > 0:
                current = candidate
        else:
            p = diff/c
            if np.isnan(p) or p >= 0:
                if diff > 0:
                    current = candidate
            elif math.exp(p) > random.random():
                current = candidate
    # Return best result
    return (current, m)


def simulated_annealing(g, max_iter=1000, init_temp=20, flip_prob=0.2,
                        cooling=cooling1):
    """Returns the size of the set D for the simulated annealing algorithm."""
    c, m = solve_simulated_annealing(
        g, max_iter, init_temp, flip_prob, cooling)
    return m


def crossover(g, a, b, p):
    """Breed function of the genetic algorithm"""
    # Compute the common set D
    mask = np.bitwise_or(a == 0, b == 0)
    new_item = np.where(mask, 0, a)

    # Choose a neighbor with probability p
    if random.random() < p:
        a = np.nonzero(mask)[0]
        if len(a) == 0:
            return new_item
        u = np.random.choice(a)
        candidate = np.copy(new_item)
        for v in g[u]:
            candidate[v] = 0
        candidate[u] = random.randint(0, 1)+1
        new_item = candidate
    return solve_greedy(g, new_item)


def solve_genetic(g, individuums, survivors, generations, mutprob):
    """Genetic algorithm for computing an inclusion-maximal bipartite
    subgraph."""

    rng = np.random.default_rng()
    # generate starting individuals
    indis = [solve_greedy(g) for x in range(individuums)]
    m = len(g)

    # Iterate over the generations
    for gen in range(generations):
        # Compute probability to reproduce
        rate = [len(g) - np.count_nonzero(x) for x in indis]
        m = min(m, min(rate))
        rate = rate - np.min(rate)
        if np.max(rate) == 0:
            rate = np.ones(individuums)
        else:
            rate = 1.1 - rate / np.max(rate)
        rate = rate / np.sum(rate)

        # Compute individuals for next generation
        next_gen = rng.choice(indis, size=survivors,
                              p=rate, replace=False).tolist()
        for y in range(individuums - survivors):
            a, b = rng.choice(indis, size=2, p=rate, replace=False)
            next_gen.append(crossover(g, a, b, mutprob))
        indis = next_gen
    rate = [len(g) - np.count_nonzero(x) for x in indis]

    # Return best element
    return min(m, min(rate))


def genetic(g, individuums=20, survivors=10, generations=200, mutprob=0.001):
    """Returns the size of the set D for the genetic algorithm."""
    return solve_genetic(g, individuums, survivors, generations, mutprob)


def investigate_file(filename):
    """Routine that runs the algorithms for all graphs in the test dataset."""

    # Parse the file
    with open(filename, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, 'xml')
    nodes = {x.get('id'): i for i, x in enumerate(bs_data.find_all('node'))}
    edges = [[nodes[x.get('source')], nodes[x.get('target')]]
             for x in bs_data.find_all('edge')]
    graph = {x: set() for x in range(len(nodes))}
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    # Check if the file is already in the database
    filename = filename.split("ata/")[1]
    with mutex:
        res = cur.execute(
            "SELECT * FROM data WHERE filename='{0}';".format(filename))
        current_line = res.fetchone()

    prin = "{0}, Order: {1}, Size: {2}\n".format(
        filename, len(graph), len(edges))

    # Execute the algorithms only, if they are not yet processed
    if current_line is None:
        with mutex:
            cur.execute(
                "INSERT INTO data (filename) VALUES ('{0}');".format(filename))
            res = cur.execute(
                "SELECT * FROM data WHERE filename='{0}';".format(filename))
            current_line = res.fetchone()
            con.commit()
    prin += ", ".join([str(x) for x in current_line]) + "\n"

    if current_line[1] is None:
        with mutex:
            cur.execute("UPDATE data SET class = '{0}' WHERE filename = '{1}';".format(
                filename.split("/")[0], filename))
            con.commit()

    if current_line[2] is None:
        with mutex:
            cur.execute("UPDATE data SET ord = {0} WHERE filename = '{1}';".format(
                len(graph), filename))
            con.commit()

    if current_line[3] is None:
        with mutex:
            cur.execute("UPDATE data SET size = {0} WHERE filename = '{1}';".format(
                len(edges), filename))
            con.commit()

    if current_line[4] is None:
        t1 = time.time()
        try:
            sat_result = sat(graph)
        except TimeoutExpired:
            sat_result = -1
        sat_time = time.time()-t1
        with mutex:
            cur.execute("UPDATE data SET sat = {0}, sat_time={1} WHERE filename = '{2}';".format(
                sat_result, sat_time, filename))
            con.commit()
        prin += "SAT: {0}, Time: {1}\n".format(sat_result, sat_time)

    if current_line[6] is None:
        t1 = time.time()
        greedy_result = greedy(graph)
        greedy_time = time.time()-t1
        with mutex:
            cur.execute("UPDATE data SET greedy = {0}, greedy_time={1} WHERE filename = '{2}';".format(
                greedy_result, greedy_time, filename))
            con.commit()
        prin += "Greedy: {0}, Time: {1}\n".format(greedy_result, greedy_time)

    if current_line[8] is None:
        t1 = time.time()
        sa_result = simulated_annealing(
            graph, max_iter=10000, init_temp=50, cooling=cooling2)
        sa_time = time.time()-t1
        with mutex:
            cur.execute("UPDATE data SET sa = {0}, sa_time={1} WHERE filename = '{2}';".format(
                sa_result, sa_time, filename))
            con.commit()
        prin += "Simulated Annealing: {0}, Time: {1}\n".format(
            sa_result, sa_time)

    if current_line[10] is None:
        t1 = time.time()
        genetic_result = genetic(
            graph, individuums=20, survivors=0, generations=10000, mutprob=1)
        genetic_time = time.time()-t1
        with mutex:
            cur.execute("UPDATE data SET genetic = {0}, genetic_time={1} WHERE filename = '{2}';".format(
                genetic_result, genetic_time, filename))
            con.commit()
        prin += "Genetic: {0}, Time: {1}\n".format(
            genetic_result, genetic_time)
    print(prin)


def test_genetic_parameters(filename):
    """A routine that allows the algorithm to parameter tune the genetic
    algorithm"""
    with open(filename, 'r') as f:
        data = f.read()

    bs_data = BeautifulSoup(data, 'xml')
    nodes = {x.get('id'): i for i, x in enumerate(bs_data.find_all('node'))}
    edges = [[nodes[x.get('source')], nodes[x.get('target')]]
             for x in bs_data.find_all('edge')]
    graph = {x: set() for x in range(len(nodes))}
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    filename = filename.split("ata/")[1]
    individuums = 20
    generations = 1000
    p = int(random.random()*11)/10
    s = genetic(graph, individuums=individuums,
                survivors=0, generations=generations, mutprob=p)
    with mutex:
        cur.execute("INSERT INTO genetic (filename,res, individuals, survivors, generations, mut_prob) VALUES ('{0}', {1}, {2}, {3}, {4},{5});".format(
            filename, s, individuums, 0, generations, p))
        con.commit()
    print(filename)


def test_sa_parameters(filename):
    """A routine that allows the algorithm to parameter tune the simulated
    annealing algorithm"""
    with open(filename, 'r') as f:
        data = f.read()

    bs_data = BeautifulSoup(data, 'xml')
    nodes = {x.get('id'): i for i, x in enumerate(bs_data.find_all('node'))}
    edges = [[nodes[x.get('source')], nodes[x.get('target')]]
             for x in bs_data.find_all('edge')]
    graph = {x: set() for x in range(len(nodes))}
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    filename = filename.split("ata/")[1]

    cooling_fun = [cooling1, cooling2, cooling3, cooling4]
    for c in range(4):

        max_iter = 10000
        init_temp = 50
        flip_prob = 0
        s = simulated_annealing(
            graph, max_iter, init_temp, flip_prob, cooling_fun[c])
        with mutex:
            cur.execute("INSERT INTO siman (filename,res,max_iter, flip_prob, init_temp, cooling) VALUES ('{0}', {1}, {2}, {3}, {4},{5});".format(
                filename, s, max_iter, flip_prob, init_temp, c))
            con.commit()
    print(filename)


def main():
    """The Routine runs all algorithms with their default parameters. If you
    want to parameter-tune uncomment the corresponding lines"""
    pool = mp.Pool(mp.cpu_count())

    jobs = []
    # Uncomment for parameter tuning of simulated annealing
    # with mutex:
    #     cur.execute("DELETE FROM siman")
    #     con.commit()

    # Uncomment for parameter tuning of genetic algorithm
    # with mutex:
    #     cur.execute("DELETE FROM genetic")
    #     con.commit()

    for i, x in enumerate(glob.iglob("./data/**/*.graphml", recursive=True)):
        job = pool.apply_async(investigate_file, (x, ))
        jobs.append(job)

        # Uncomment for parameter tuning of simulated annealing
        # job = pool.apply_async(test_genetic_parameters, (x,))
        # jobs.append(job)

        # Uncomment for parameter tuning of genetic algorithm
        # job = pool.apply_async(test_sa_parameters, (x,))
        # jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    pool.close()
    pool.join()
    con.close()


if __name__ == "__main__":
    main()
