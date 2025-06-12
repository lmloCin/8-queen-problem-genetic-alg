import random

def decimal_to_binary_string(number, num_bits=3):
    """Convert a decimal number to a binary string."""
    if not (0 <= number <= 7):
        raise ValueError("Number must be between 0 and 7")
    return format(number, f"0{str(num_bits)}b")

def permutation_to_bit_string(permutation_list):
    """Convert a permutation list to a bit string."""
    bit_parts = [decimal_to_binary_string(num) for num in permutation_list]
    return "".join(bit_parts)

def bit_string_to_permutation(bit_string, num_bits_per_gene=3):
    """Convert a bit string to a permutation list of integers."""
    permutation = []
    # Iterates over the bit string in chunks of num_bits_per_gene
    for i in range(0, len(bit_string), num_bits_per_gene):
        gene_bits = bit_string[i : i + num_bits_per_gene]
        permutation.append(int(gene_bits, 2))
    return permutation

def create_fathers():
    father = ''
    p_vals = ['000', '001', '010', '011', '100', '101', '110', '111']
    for i in range(8):
        random_num = random.choice(p_vals)
        p_vals.remove(random_num)
        father += random_num

    return father


def fitness(ind):
    fitness = 0
    for i in range(len(ind)):
        col = ind[i]
        for j in range(len(ind)):
            # Don't compare with itself
            if j == i:
                continue
            # Attacks in the same line, but in practice, never reaches this
            if ind[j] == col:
                continue
            # Secondary diagonal
            if j + ind[j] == i + col:
                continue
            # Main diagonal
            if j - ind[j] == i - col:
                continue
            fitness += 1
    return fitness / 2

# Swap Mutation
def mutation(ind, prob):
    if random.randrange(1, 101) <= prob :
        random1 = random.choice(ind)
        index1 = ind.index(random1)
        ind.remove(random1)
        random2 = random.choice(ind)
        index2 = ind.index(random2)
        ind.insert( index2, random1)
        ind.remove(random2)
        ind.insert( index1, random2)

    return ind

# Cut-and-Cossfill crossover
def cut_crossfill_crossover(father, mother, prob):
    if random.randrange(1, 101) <= prob :
        n = len(father)
        son1 = [None] * n
        son2 = [None] * n

        # Choosing two random cut points
        cut1 = random.randrange(0, n-1)
        cut2 = random.randrange(cut1 + 1, n)

        # Copy central segment of father to son1 and mother to son2
        for i in range(cut1, cut2 + 1):
            son1[i] = father[i]
            son2[i] = mother[i]

        # Fill the rest of the son1 with the mother's values
        mother_genes_fill = []
        mother_scan_start_idx = (cut2 + 1) % n

        # Sweep through mother's genes and fill the rest of the son1
        for _ in range(n):
            gene = mother[mother_scan_start_idx]

            if gene not in son1[cut1:cut2 + 1]:
                mother_genes_fill.append(gene)
            
            mother_scan_start_idx = (mother_scan_start_idx + 1) % n

        # Put the genes in the right order in son1
        son1_fill_idx = (cut2 + 1) % n
        for gene in mother_genes_fill:
            while son1[son1_fill_idx] is not None:
                son1_fill_idx = (son1_fill_idx + 1) % n
            son1[son1_fill_idx] = gene

        # Fill the rest of the son2 with the father's values
        father_genes_fill = []
        father_scan_start_idx = (cut2 + 1) % n

        # Sweep through father's genes and fill the rest of the son2
        for _ in range(n):
            gene = father[father_scan_start_idx]

            if gene not in son2[cut1:cut2 + 1]:
                father_genes_fill.append(gene)
            
            father_scan_start_idx = (father_scan_start_idx + 1) % n

        # Put the genes in the right order in son2
        son2_fill_idx = (cut2 + 1) % n
        for gene in father_genes_fill:
            while son2[son2_fill_idx] is not None:
                son2_fill_idx = (son2_fill_idx + 1) % n
            son2[son2_fill_idx] = gene
        
    else:
        son1 = list(father)
        son2 = list(mother)
    return son1, son2


def recombination(father, mother, prob):
    son1 = []
    son2 = []
    if random.randrange(1, 101) <= prob :
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
# Creating aleatory initial population of 100 individuals
population = 100
for i in range(population):
    ind = create_fathers()
    ind = bit_string_to_permutation(ind)
    possible_solves.append(ind)
    fitness_solves.append(fitness(ind))
    fitness_count += 1

avg_fit = sum(fitness_solves)/population
best_fit = max(fitness_solves)
best_sol = possible_solves[fitness_solves.index(best_fit)]
print(f"Fitness médio_inicial: {avg_fit}\nFitness máximo_inicial: {best_fit}")


interactions = 0
# End condition: find solution or 10000 fitness evaluations
while best_fit < 28 and fitness_count < 10000:
    interactions += 1
    sons = []
    sons_fit = []
    possibles_parents= []
    possibles_parents_fitness = []

    # Generate 2 sons for each generation
    for j in range(0, 5):
        # Tournament selection of 5 individuals
        random_factor = random.randrange(0, 100)
        possibles_parents.append(possible_solves[random_factor])
        possibles_parents_fitness.append(fitness_solves[random_factor])
    # Pick the best two parents
    father = possibles_parents[possibles_parents_fitness.index(max(possibles_parents_fitness))]
    possibles_parents_fitness.remove(max(possibles_parents_fitness))
    possibles_parents.remove(father)
    mother = possibles_parents[possibles_parents_fitness.index(max(possibles_parents_fitness))]
    possibles_parents_fitness.remove(max(possibles_parents_fitness))
    possibles_parents.remove(mother)

    # Recombination probability = 90%
    son1, son2 = cut_crossfill_crossover(father, mother, 90)
    # Mutation probability = 40%
    son1, son2 = mutation(son1, 40), mutation(son2, 40)

    sons.append(son1)
    sons_fit.append(fitness(son1))
    sons.append(son2)
    sons_fit.append(fitness(son2))
    fitness_count += 2

 
    # Replace the worst individuals with the new individuals
    for i in range(2):
        worst_ind_fitness = min(fitness_solves)
        worst_ind_index = fitness_solves.index(worst_ind_fitness)

        fitness_solves.pop(worst_ind_index)
        possible_solves.pop(worst_ind_index)

    for i in range(2):
        possible_solves.append(sons[i])
        fitness_solves.append(sons_fit[i])
    
    # Update best fitness
    avg_fit = sum(fitness_solves)/population
    best_fit = max(fitness_solves)  
    best_sol = possible_solves[fitness_solves.index(best_fit)]
    print(f"\nInteraction: {interactions} \nActual_Avg_Fitness : {avg_fit}\nActual_Best_Fitness: {best_fit}")

            
avg_fit = sum(fitness_solves)/population
best_fit = max(fitness_solves)
best_sol = possible_solves[fitness_solves.index(best_fit)]
best_sol_bit_string = permutation_to_bit_string(best_sol)
print(f'\nInteractions number: {interactions} \nFitness_count: {fitness_count} \nBest_solution: {best_sol} \nBest_fitness: {best_fit} \nAvg_end_fitness: {avg_fit} \nBest_solution_bit_string: {best_sol_bit_string}')
print_board(best_sol)