import random


def create_fathers():
    prob_solve = []
    p_vals = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(8):
        random_num = random.choice(p_vals)
        p_vals.remove(random_num)
        prob_solve.append(random_num)

    return prob_solve


def fitness(ind):
    fitness = 0
    for i in range(len(ind)):
        col = ind[i]
        for j in range(len(ind)):
            if j == i:
                continue
            if ind[j] == col:
                continue
            if j + ind[j] == i + col:
                continue
            if j - ind[j] == i - col:
                continue
            fitness += 1
    return fitness / 2


def mutation(ind, prob):
    if random.randrange(1, 100) <=prob :
        random1 = random.choice(ind)
        index1 = ind.index(random1)
        ind.remove(random1)
        random2 = random.choice(ind)
        index2 = ind.index(random2)
        ind.insert( index2, random1)
        ind.remove(random2)
        ind.insert( index1, random2)

    return ind



def recombination(father, mother, prob):
    son1 = []
    son2 = []
    if random.randrange(1, 100) <=prob :
        for i in range(3):
            son1.append(father[i])
            son2.append(mother[i])
        for i in range(5):
            if not(mother[3 + i] in son1):
                son1.append(mother[3 + i])
            if not(father[3 + i] in son2):
                son2.append(father[3 + i])
        if len(son1) != 8:
            for i in range(3):
                if not(mother[i] in son1):
                    son1.append(mother[i])
        if len(son2) != 8:
            for i in range(3):
                if not(father[i] in son2):
                    son2.append(father[i])
    else:
        son1 = father
        son2 = mother
    return son1, son2


def recombination2(father, mother):
    son1 = []
    son2 = []
    for i in range(2):
        son1.append(father[i])
        son2.append(mother[i])
    for i in range(6):
        if not(mother[2 + i] in son1):
            son1.append(mother[2 + i])
        if not(father[2 + i] in son2):
            son2.append(father[2 + i])
    if len(son1) != 8:
         for i in range(2):
            if not(mother[i] in son1):
                son1.append(mother[i])
    if len(son2) != 8:
         for i in range(2):
            if not(father[i] in son2):
                son2.append(father[i])
    return son1, son2


def print_board(board):
    n = len(board)
    for row in range(n):
        line = ""
        for col in range(n):
            if board[row] == col:
                line += "Q "
            else:
                line += "- "
        print(line)
    print()


fitness_count = 0
possible_solves = []
fitness_solves = []
#criando papais e mamaes
population = 100
for i in range(population):
    ind = create_fathers()
    possible_solves.append(ind)
    fitness_solves.append(fitness(ind))
    fitness_count += 1

avg_fit = sum(fitness_solves)/population
best_fit = max(fitness_solves)
best_sol = possible_solves[fitness_solves.index(best_fit)]
print(f"Fitness médio_inicial: {avg_fit}\nFitness máximo_inicial: {best_fit}")


interactions = 0
while best_fit < 28 and fitness_count < 10000:
    interactions += 1
    sons = []
    sons_fit = []
    possibles_parents= []
    possibles_parents_fitness = []
    for i in range(population):
        for j in range(0, 5):
            random_factor = random.randrange(0, 100)
            possibles_parents.append(possible_solves[random_factor])
            possibles_parents_fitness.append(fitness_solves[random_factor])
        father = possibles_parents[possibles_parents_fitness.index(max(possibles_parents_fitness))]
        mother = possibles_parents[possibles_parents_fitness.index(max(possibles_parents_fitness))]
        son1, son2 = recombination(father, mother, 90)
        son1, son2 = mutation(son1, 40), mutation(son2, 40)
        sons.append(son1)
        sons_fit.append(fitness(son1))
        sons.append(son2)
        sons_fit.append(fitness(son2))
        fitness_count += 2

 
    for i in range(population): 
        sons_fit.append(fitness_solves[i])
        sons.append(possible_solves[i])
    fitness_solves = []
    possible_solves = []
    for i in range(population):
        sel_son_fit = max(sons_fit)
        selected_son = sons.pop(sons_fit.index(sel_son_fit))
        sons_fit.remove(sel_son_fit)
        possible_solves.append(selected_son)
        fitness_solves.append(sel_son_fit)
    best_fit = max(fitness_solves)
    best_sol = possible_solves[fitness_solves.index(best_fit)]
            

avg_fit = sum(fitness_solves)/population
best_fit = max(fitness_solves)
best_sol = possible_solves[fitness_solves.index(best_fit)]
print(f'best_solution: {best_sol} \nbest_fitness: {best_fit} \ninteractions_number: {interactions} \navg_end_fitness: {avg_fit}\nFitness_count:{fitness_count}')
#print_board(best_sol)
