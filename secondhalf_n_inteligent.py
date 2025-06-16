import random
import matplotlib.pyplot as plt
import math # Import math for sqrt and potentially other functions

# --- GLOBAL PARAMETERS FOR N-QUEENS PROBLEM ---
# Define N_QUEENS here. Change this value to run for different N.
N_QUEENS = 12
# Calculate the number of bits required for each gene (row position 0 to N-1)
NUM_BITS_PER_GENE = (N_QUEENS - 1).bit_length() if N_QUEENS > 1 else 1
# Calculate the maximum possible fitness (N choose 2)
MAX_FITNESS = (N_QUEENS * (N_QUEENS - 1)) / 2 if N_QUEENS > 1 else 0

# Handle N=1 case specifically
if N_QUEENS == 1:
    MAX_FITNESS = 0
    NUM_BITS_PER_GENE = 1 # A single bit '0' is enough for 0-indexed position 0
elif N_QUEENS < 1:
    raise ValueError("N_QUEENS must be at least 1.")


# --- Binary Conversion Functions (Generalized) ---
def decimal_to_binary_string(number, num_bits):
    """
    Convert a decimal number to a binary string with a specified number of bits.
    Adjusted to handle any valid number within the bit range for N_QUEENS.
    """
    # Check if the number can be represented with the given num_bits
    if not (0 <= number < (1 << num_bits)):
        raise ValueError(f"Number {number} out of range for {num_bits} bits (0 to {(1 << num_bits) - 1})")
    return format(number, f"0{num_bits}b")

def permutation_to_bit_string(permutation_list, num_bits_per_gene_val):
    """
    Convert a permutation list (integers) to a bit string.
    Takes num_bits_per_gene_val as argument.
    """
    bit_parts = [decimal_to_binary_string(num, num_bits_per_gene_val) for num in permutation_list]
    return "".join(bit_parts)

def bit_string_to_permutation(bit_string, num_bits_per_gene_val):
    """
    Convert a bit string to a permutation list of integers.
    Takes num_bits_per_gene_val as argument.
    """
    permutation = []
    # Iterates over the bit string in chunks of num_bits_per_gene_val
    for i in range(0, len(bit_string), num_bits_per_gene_val):
        gene_bits = bit_string[i : i + num_bits_per_gene_val]
        # Ensure gene_bits is not empty if bit_string length isn't a perfect multiple
        if gene_bits:
            permutation.append(int(gene_bits, 2))
    return permutation

# --- Chromosome Creation Functions (Generalized) ---
def create_fathers_improved(n_queens_val, num_bits_per_gene_val):
    """
    Cria uma permutação aleatória de N rainhas como uma string binária.
    A lógica de filtragem de diagonais adjacentes é um heurística de inicialização.
    """
    while True:  # Loop to ensure a valid permutation is always generated
        available_rows = list(range(n_queens_val))
        father_as_list = []  # Stores the queen positions (as integers from 0 to n_queens_val-1)

        # Attempt to place all N queens
        for column in range(n_queens_val):
            # Get the position of the last placed queen (if it's not the first column)
            last_row = father_as_list[-1] if column > 0 else -100 # Dummy value that won't cause conflict

            # Filter available rows to remove those that cause an *adjacent* diagonal attack.
            # This is a specific heuristic for initial population.
            valid_options = [
                r for r in available_rows
                if abs(r - last_row) != 1
            ]

            # If no valid options, this attempt failed, break and try to build a new one
            if not valid_options:
                break

            chosen_row = random.choice(valid_options)
            father_as_list.append(chosen_row)
            available_rows.remove(chosen_row)

        # If the list has N queens, the construction was successful
        if len(father_as_list) == n_queens_val: # Check against N_QUEENS
            # Convert the list of integers to the final binary string format
            final_father = ''.join([format(pos, f'0{num_bits_per_gene_val}b') for pos in father_as_list])
            return final_father

# --- Fitness Function (Already N-compatible) ---
def fitness(ind):
    """Calculates the fitness of an individual (number of non-attacking pairs)."""
    fitness_score = 0
    n = len(ind) # N is implicitly the number of queens/columns
    for i in range(n):
        col_i_val = ind[i] # Row position of queen in column i
        for j in range(n):
            if j == i:
                continue # Don't compare with itself
            
            col_j_val = ind[j] # Row position of queen in column j

            # Check for attacks
            if col_j_val == col_i_val: # Same row (redundant for permutation encoding, but harmless)
                continue
            if abs(j - i) == abs(col_j_val - col_i_val): # Diagonal attack
                continue
            
            # If no attack, they don't attack each other
            fitness_score += 1
    return fitness_score / 2

# --- Mutation and Crossover Functions (Already N-compatible) ---
# Swap Mutation
def mutation(ind_list, prob): # Renamed 'ind' to 'ind_list' for clarity
    """Applies swap mutation to an individual's permutation list."""
    if random.randrange(1, 101) <= prob :
        if len(ind_list) < 2: # Need at least two elements to swap
            return ind_list

        idx1, idx2 = random.sample(range(len(ind_list)), 2)
        ind_list[idx1], ind_list[idx2] = ind_list[idx2], ind_list[idx1]
    return ind_list

# Intelligent Mutation (Generalized)
def calculate_attacks(individual_list): # Renamed 'individual' to 'individual_list' for clarity
    """
    Calculates the number of diagonal attacks for a given individual list.
    """
    attacks = 0
    n = len(individual_list)
    for i in range(n):
        for j in range(i + 1, n): # Only check each pair once
            if abs(i - j) == abs(individual_list[i] - individual_list[j]):
                attacks += 1
    return attacks

def intelligent_mutation(individual_list, mutation_prob): # Renamed 'individual' to 'individual_list'
    """
    Performs an "intelligent" mutation.
    The swap is only accepted if it does not increase the number of diagonal attacks.
    """
    if random.uniform(0, 1) < mutation_prob:
        original_attacks = calculate_attacks(individual_list)

        # If it's already a perfect solution, no need to mutate
        if original_attacks == 0:
            return individual_list

        mutated_individual_list = list(individual_list) # Create a copy
        
        if len(mutated_individual_list) < 2: # Need at least two elements to swap
            return individual_list

        idx1, idx2 = random.sample(range(len(mutated_individual_list)), 2)
        mutated_individual_list[idx1], mutated_individual_list[idx2] = mutated_individual_list[idx2], mutated_individual_list[idx1]

        new_attacks = calculate_attacks(mutated_individual_list)

        if new_attacks <= original_attacks:
            return mutated_individual_list
        else:
            return individual_list
    return individual_list

# Cut-and-Crossfill crossover
def cut_crossfill_crossover(father_list, mother_list, prob): # Renamed father/mother for clarity
    """Applies cut-and-crossfill crossover to two parents' permutation lists."""
    if random.randrange(1, 101) <= prob :
        n = len(father_list)
        son1 = [None] * n
        son2 = [None] * n

        if n < 2: # Cannot perform crossover if n is less than 2
            return list(father_list), list(mother_list)

        # Choosing two random cut points
        cut1 = random.randrange(0, n - 1)
        cut2 = random.randrange(cut1 + 1, n)

        # Copy central segment of father to son1 and mother to son2
        for i in range(cut1, cut2 + 1):
            son1[i] = father_list[i]
            son2[i] = mother_list[i]

        # Fill the rest of son1 with mother's values
        mother_genes_fill = []
        # Create a scan order for mother's genes, wrapping around
        mother_scan_order = mother_list[cut2 + 1:] + mother_list[:cut2 + 1]

        for gene in mother_scan_order:
            if gene not in son1[cut1:cut2 + 1]:
                mother_genes_fill.append(gene)
        
        son1_fill_idx = (cut2 + 1) % n
        for gene in mother_genes_fill:
            while son1[son1_fill_idx] is not None:
                son1_fill_idx = (son1_fill_idx + 1) % n
            son1[son1_fill_idx] = gene

        # Fill the rest of son2 with father's values
        father_genes_fill = []
        # Create a scan order for father's genes, wrapping around
        father_scan_order = father_list[cut2 + 1:] + father_list[:cut2 + 1]

        for gene in father_scan_order:
            if gene not in son2[cut1:cut2 + 1]:
                father_genes_fill.append(gene)
        
        son2_fill_idx = (cut2 + 1) % n
        for gene in father_genes_fill:
            while son2[son2_fill_idx] is not None:
                son2_fill_idx = (son2_fill_idx + 1) % n
            son2[son2_fill_idx] = gene
        
    else:
        son1 = list(father_list)
        son2 = list(mother_list)
    return son1, son2


# --- Removed unused recombination functions (recombination, recombination2) ---


# --- Utility Functions (Generalized / Corrected) ---
def print_board(board):
    """Prints the N-Queens board configuration."""
    n = len(board)
    for row_idx in range(n):
        line = ""
        for col_idx in range(n):
            # board[col_idx] gives the row for that column
            if board[col_idx] == row_idx:
                line += "Q "
            else:
                line += "- "
        print(line)
    print()


def standart_deviation(indv):
    """
    Calculates the standard deviation of a list of numbers.
    Corrected for sample standard deviation and edge cases.
    """
    if len(indv) <= 1:
        return 0.0 # Standard deviation is 0 or undefined for 0 or 1 element
    
    mean = sum(indv) / len(indv)
    # Using len(indv) - 1 for sample standard deviation (unbiased estimator)
    variance = sum([(x - mean) ** 2 for x in indv]) / (len(indv) - 1)
    return math.sqrt(variance)

# Selection: Binary Tournament (Already N-compatible)
def selection(population, fitnesses):
    """Selects individuals using binary tournament."""
    selected = []
    # We want to select `len(population)` individuals for the next generation
    for _ in range(len(population)):
        if len(population) < 2:
            return population # Cannot perform tournament with less than 2 individuals
        
        idx1, idx2 = random.sample(range(len(population)), 2)
        winner = population[idx1] if fitnesses[idx1] > fitnesses[idx2] else population[idx2]
        selected.append(winner)
    return selected


# --- Main Experiment Function (Generalized) ---
def run_single_experiment(n_queens_val, max_fitness_val, num_bits_per_gene_val):
    fitness_count = 0
    possible_solves = [] # Stores individuals as permutation lists (integers)
    fitness_solves = []
    population_size = 100 # Standard population size
    true_solves = 0 # Count of individuals that reached optimal fitness in the final population
    
    # Creating initial population of 'population_size' individuals
    for i in range(population_size):
        # `create_fathers_improved` now returns a bit string based on n_queens_val
        ind_bit_string = create_fathers_improved(n_queens_val, num_bits_per_gene_val)
        # Convert to permutation list for fitness evaluation and genetic operations
        ind_permutation = bit_string_to_permutation(ind_bit_string, num_bits_per_gene_val)
        
        possible_solves.append(ind_permutation)
        fitness_solves.append(fitness(ind_permutation))
        fitness_count += 1

    avg_fit = sum(fitness_solves) / population_size
    best_fit = max(fitness_solves)
    best_sol = possible_solves[fitness_solves.index(best_fit)]
    print(f"Fitness médio inicial: {avg_fit:.2f}\nFitness máximo inicial: {best_fit:.2f}")


    interactions = 0
    # End condition: find solution or max fitness evaluations (e.g., 10000)
    while best_fit < max_fitness_val and fitness_count < 10000:
        interactions += 1
        sons = []
        sons_fit = []
        
        # Generational Replacement: Create 'population_size' new individuals
        for _ in range(population_size // 2):
            # Select two parents using binary tournament
            parents_for_crossover = selection(possible_solves, fitness_solves)
            father_perm, mother_perm = parents_for_crossover[0], parents_for_crossover[1]

            # Crossover probability = 85%
            son1_perm, son2_perm = cut_crossfill_crossover(father_perm, mother_perm, 85)
            
            # Mutation probability = 40%
            son1_perm = intelligent_mutation(son1_perm, 0.40) # Use intelligent_mutation with a float probability
            son2_perm = intelligent_mutation(son2_perm, 0.40)

            sons.append(son1_perm)
            sons_fit.append(fitness(son1_perm))
            sons.append(son2_perm)
            sons_fit.append(fitness(son2_perm))
            fitness_count += 2 # Each crossover generates two individuals, so 2 fitness evaluations

        # Replace the entire old generation with the new one
        possible_solves = sons
        fitness_solves = sons_fit
        
        # Update best fitness and average fitness for monitoring
        avg_fit = sum(fitness_solves) / population_size
        best_fit = max(fitness_solves)  
        best_sol = possible_solves[fitness_solves.index(best_fit)]
       # print(f"\nInteraction: {interactions} \nActual_Avg_Fitness : {avg_fit:.2f}\nActual_Best_Fitness: {best_fit:.2f}")

                
    # Final results for this single experiment run
    avg_fit = sum(fitness_solves) / population_size
    best_fit = max(fitness_solves)
    best_sol = possible_solves[fitness_solves.index(best_fit)]
    # Convert best solution back to bit string for display if needed
    best_sol_bit_string = permutation_to_bit_string(best_sol, num_bits_per_gene_val)

    # Count the number of boards that converged to the optimal fitness in the final population
    for f in fitness_solves: # Use 'f' to avoid conflict with function 'fitness'
        if int(f) == int(max_fitness_val): # Compare as integers
            true_solves += 1

    # Return results for analysis
    converged = (int(best_fit) == int(max_fitness_val)) # Compare as integers
    return {
        "converged": converged,
        "interactions": interactions,
        "fitness_count": fitness_count,
        "final_best_fitness": best_fit,
        "final_best_solution": best_sol,
        "final_avg_fitness": avg_fit,
        "final_board_converged": true_solves
    }

# ---- Main Analysis Loop ----
num_executions = 30 # Number of times to run the experiment
convergence_count = 0
total_interactions = []
total_fitness_evals = 0
all_final_best_fitnesses = []
all_final_avg_fitnesses = []
best_solution_overall = None
best_fitness_overall = -1
total_final_boards_converged = 0

print(f"Starting {num_executions} executions for N={N_QUEENS}...\n")

for exec_num in range(1, num_executions + 1):
    print(f"--- Execution {exec_num}/{num_executions} (N={N_QUEENS}) ---")
    # Pass global N_QUEENS, MAX_FITNESS, and NUM_BITS_PER_GENE
    results = run_single_experiment(N_QUEENS, MAX_FITNESS, NUM_BITS_PER_GENE)

    if results["converged"]:
        convergence_count += 1
        print(f"  Converged! Interactions: {results['interactions']}, Fitness Evals: {results['fitness_count']}")
    else:
        print(f"  Did NOT converge. Interactions: {results['interactions']}, Fitness Evals: {results['fitness_count']}")

    total_interactions.append(results["interactions"])
    total_fitness_evals += results["fitness_count"]
    all_final_best_fitnesses.append(results["final_best_fitness"])
    all_final_avg_fitnesses.append(results["final_avg_fitness"])
    total_final_boards_converged += results["final_board_converged"]

    if results["final_best_fitness"] > best_fitness_overall:
        best_fitness_overall = results["final_best_fitness"]
        best_solution_overall = results["final_best_solution"]

print("\n" + "="*50)
print(f"--- Overall Analysis Results for N={N_QUEENS} ---")
print(f"Total executions: {num_executions}")
print(f"Executions converged: {convergence_count} / {num_executions}")
print(f"Convergence rate: {convergence_count / num_executions * 100:.2f}%")
print(f"Average final best fitness: {sum(all_final_best_fitnesses) / num_executions:.2f}")
print(f"Average final fitness: {sum(all_final_avg_fitnesses) / num_executions:.2f}")
print(f"Average Fitness standard deviation: {standart_deviation(all_final_avg_fitnesses):.2f}")
print(f"Average interactions for all runs: {sum(total_interactions) / num_executions:.2f}")
print(f"Interactions standard deviation: {standart_deviation(total_interactions):.2f}")
print(f"Average fitness evaluations for all runs: {total_fitness_evals / num_executions:.2f}")
print(f"Total boards that converged: {total_final_boards_converged}")
print(f"Average boards that converged: {total_final_boards_converged/num_executions:.2f}")
print(f"Overall best fitness found: {best_fitness_overall:.2f}")
if best_solution_overall:
    print(f"Overall best solution (decimal): {best_solution_overall}")
    # Use NUM_BITS_PER_GENE when converting to bit string for display
    print(f"Overall best solution (bit string): {permutation_to_bit_string(best_solution_overall, NUM_BITS_PER_GENE)}")
    print("\nBoard for overall best solution:")
    print_board(best_solution_overall)
print("="*50)

# --- Plotting Results ---
x = list(range(1, num_executions + 1))
sum_avg = [sum(all_final_avg_fitnesses) / num_executions] * num_executions
sum_int = [sum(total_interactions) / num_executions] * num_executions

# Prepare text for display on the plot
text_summary = (f"--- Overall Analysis Results for N={N_QUEENS} ---\n"
f"Total executions: {num_executions}\n"
f"Executions converged: ({convergence_count} / {num_executions})\n"
f"Convergence rate: {convergence_count / num_executions * 100:.2f}%\n"
f"Average final best fitness: {sum(all_final_best_fitnesses) / num_executions:.2f}\n"
f"Average final fitness: {sum(all_final_avg_fitnesses) / num_executions:.2f}\n"
f"Average Fitness standard deviation: {standart_deviation(all_final_avg_fitnesses):.2f}\n"
f"Average interactions for all runs: {sum(total_interactions) / num_executions:.2f}\n"
f"Interactions standard deviation: {standart_deviation(total_interactions):.2f}\n"
f"Average fitness evaluations for all runs: {total_fitness_evals / num_executions:.2f}\n"
f"Total boards that converged: {total_final_boards_converged}\n"
f"Average boards that converged: {total_final_boards_converged/num_executions:.2f}\n"
f"Overall best fitness found: {best_fitness_overall:.2f}")

# First graph: runs vs all final average fitness
plt.figure(figsize = (15, 8)) # Increased figure size for better readability
plt.subplot(1, 2, 1)
plt.title(f'Runs vs. Final Average Fitness (N={N_QUEENS})')
plt.plot(x, all_final_avg_fitnesses, label='Individual Run Final Average Fitness')
plt.plot(x, sum_avg, label='Overall Average Final Fitness', linestyle='--')
plt.xlabel('Run Number')
plt.ylabel('Average Fitness')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend() # Added legend

# Second graph: runs vs all final interactions
plt.subplot(1, 2, 2)
plt.title(f"Runs vs. Total Interactions (N={N_QUEENS})", fontsize = 16)
plt.plot(x, total_interactions, label='Individual Run Total Interactions')
plt.plot(x, sum_int, label='Overall Average Interactions', linestyle='--')
plt.xlabel('Run Number')
plt.ylabel('Interactions')
plt.legend() # Added legend
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplots_adjust(bottom=0.35, wspace=0.3) # Adjusted bottom and wspace for text and spacing

plt.figtext(0.5, 0.05, text_summary, ha="center", fontsize=10, 
            bbox={"facecolor":"lightgray", "alpha":0.5, "pad":6})

plt.show()

print("="*50)