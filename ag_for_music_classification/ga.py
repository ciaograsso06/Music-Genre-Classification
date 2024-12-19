import json
import random
import numpy as np
from deap import base, creator, tools, algorithms

with open('training_cfg.json', 'r') as f:
    original_config = json.load(f)

def evaluate_model(config):
    return random.uniform(0.5, 1.0),

def create_individual():
    return {
        "output_dir": "./results",
        "num_epochs": random.randint(1, 100),
        "lr": random.uniform(0.00001, 0.01),
        "name": "genetic_exp",
        "per_device_train_batch_size": random.randint(8, 128),
        "per_device_eval_batch_size": random.randint(32, 256),
        "warmup_steps": random.randint(100, 1000),
        "weight_decay": random.uniform(0.0, 0.1),
        "logging_dir": "./logs"
    }

def mutate_individual(individual):
    key = random.choice(list(individual.keys()))
    if key == "num_epochs":
        individual[key] = random.randint(1, 100)
    elif key == "lr":
        individual[key] = random.uniform(0.00001, 0.01)
    elif key in ["per_device_train_batch_size", "per_device_eval_batch_size"]:
        individual[key] = random.randint(8, 128) if key == "per_device_train_batch_size" else random.randint(32, 256)
    elif key == "warmup_steps":
        individual[key] = random.randint(100, 1000)
    elif key == "weight_decay":
        individual[key] = random.uniform(0.0, 0.1)

def crossover(ind1, ind2):
    key = random.choice(list(ind1.keys()))
    ind1[key], ind2[key] = ind2[key], ind1[key]

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_model)

population_size = 20
generations = 10
mutation_probability = 0.2
crossover_probability = 0.5

population = toolbox.population(n=population_size)

for gen in range(generations):
    print(f"Generation {gen}")
    
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))
    
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < crossover_probability:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    
    for mutant in offspring:
        if random.random() < mutation_probability:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    population[:] = offspring

best_individual = tools.selBest(population, 1)[0]
print("Best configuration:", best_individual)
