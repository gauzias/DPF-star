import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random

# Chargement des données
dev_df = pd.read_csv("stats_dev.csv")
std_df = pd.read_csv("stats_std.csv")
diff_df = pd.read_csv("stats_diff.csv")

# Colonnes alpha
alpha_cols = ['0', '1', '5', '10', '50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '800', '900', '1000', '2000']
alpha_indices = list(range(len(alpha_cols)))

# Création du problème d'optimisation multi-objectifs
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))  # Minimize dev & std, maximize diff
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.choice, alpha_indices)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fonction d'évaluation
def evaluate(individual):
    idx = individual[0]
    col = alpha_cols[idx]
    dev_mean = dev_df[col].mean()
    std_mean = std_df[col].mean()
    diff_mean = diff_df[col].mean()
    return dev_mean, std_mean, diff_mean

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(alpha_cols)-1, indpb=1.0)
toolbox.register("select", tools.selNSGA2)

# Paramètres NSGA-II
population = toolbox.population(n=50)
NGEN = 40
CXPB = 0.7
MUTPB = 0.2

# Évaluer la population initiale
for ind in population:
    ind.fitness.values = toolbox.evaluate(ind)

# Algorithme principal
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)

    # Évaluer les fitness des nouveaux individus
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    for ind in invalid_ind:
        ind.fitness.values = toolbox.evaluate(ind)

    # Fusionner avec la population actuelle
    combined = population + offspring

    # S'assurer que tous les individus ont une fitness
    for ind in combined:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    # Sélection NSGA-II
    population = toolbox.select(combined, k=len(population))

# Récupérer le front de Pareto
pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

# Affichage des résultats
print("Solutions Pareto optimales (alpha, dev, std, diff):")
for ind in pareto_front:
    i = ind[0]
    dev, std, diff = evaluate(ind)
    print(f"alpha={alpha_cols[i]}, dev={dev:.4f}, std={std:.4f}, diff={diff:.4f}")

# Supprimer les doublons de alpha
unique_alphas = set()
filtered_front = []
for ind in pareto_front:
    alpha = alpha_cols[ind[0]]
    if alpha not in unique_alphas:
        filtered_front.append(ind)
        unique_alphas.add(alpha)

print("Solutions Pareto optimales uniques (alpha, dev, std, diff):")
for ind in filtered_front:
    i = ind[0]
    dev, std, diff = evaluate(ind)
    print(f"alpha={alpha_cols[i]}, dev={dev:.4f}, std={std:.4f}, diff={diff:.4f}")
