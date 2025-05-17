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
    for i in range(7):
        if (ind[i+ 1] == ind[i] + 1) or (ind[i+ 1] == ind[i] - 1):
            fitness += 1
    #return (1/(1 + fitness))
    return fitness


def fitness2(ind):
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


def mutation(ind):
    if random.randrange(1, 10) <=9 :
        random1 = random.choice(ind)
        index1 = ind.index(random1)
        ind.remove(random1)
        random2 = random.choice(ind)
        index2 = ind.index(random2)
        ind.insert( index2, random1)
        ind.remove(random2)
        ind.insert( index1, random2)

    return ind



def recombination(father, mother):
    son1 = []
    son2 = []
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

possible_solves = []
fitness_solves = []
#criando papais e mamaes
for i in range(10):
    ind = create_fathers()
    possible_solves.append(ind)
    fitness_solves.append(fitness2(ind))


best_fit = max(fitness_solves)
best_sol = possible_solves[fitness_solves.index(best_fit)]
interactions = 0
while best_fit < 28 :
    interactions += 1
    sons = []
    sons_fit = []
    for i in range(10):
        chance = random.randrange(1,5)
        if chance == 1:
            father = possible_solves[random.randrange(0, 2)]
            mother = possible_solves[random.randrange(0, 2)]
            son1, son2 = recombination(father, mother)
            son1, son2 = mutation(son1), mutation(son2)
            sons.append(son1)
            sons_fit.append(fitness2(son1))
            sons.append(son2)
            sons_fit.append(fitness2(son2))
        elif chance == 2:
            father = possible_solves[random.randrange(0, 4)]
            mother = possible_solves[random.randrange(0, 4)]
            son1, son2 = recombination(father, mother)
            son1, son2 = mutation(son1), mutation(son2)
            sons.append(son1)
            sons_fit.append(fitness2(son1))
            sons.append(son2)
            sons_fit.append(fitness2(son2))
        elif chance == 3:
            father = possible_solves[random.randrange(0, 6)]
            mother = possible_solves[random.randrange(0, 6)]
            son1 , son2 = recombination(father, mother)
            son1, son2 = mutation(son1), mutation(son2)
            sons.append(son1)
            sons_fit.append(fitness2(son1))
            sons.append(son2)
            sons_fit.append(fitness2(son2))
        elif chance == 4:
            father = possible_solves[random.randrange(0, 8)]
            mother = possible_solves[random.randrange(0, 8)]
            son1, son2 = recombination(father, mother)
            son1, son2 = mutation(son1), mutation(son2)
            sons.append(son1)
            sons_fit.append(fitness2(son1))
            sons.append(son2)
            sons_fit.append(fitness2(son2))
    possible_solves = []
    fitness_solves = []
    for i in range(10):
        sel_son_fit = max(sons_fit)
        #print(sel_son_fit)
        selected_son = sons.pop(sons_fit.index(sel_son_fit))
        #print(selected_son)
        sons_fit.remove(sel_son_fit)
        possible_solves.append(selected_son)
        fitness_solves.append(sel_son_fit)
    best_fit = max(fitness_solves)
    best_sol = possible_solves[fitness_solves.index(best_fit)]
            

#for i in range(10):
#    print(possible_solves[i], '', fitness_solves[i])

best_fit = max(fitness_solves)
best_sol = possible_solves[fitness_solves.index(best_fit)]
print('\n' , best_sol, best_fit, 'interections:', interactions)
print_board(best_sol)
