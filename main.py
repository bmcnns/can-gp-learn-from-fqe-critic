import pickle
from copy import deepcopy
from statistics import median
from uuid import uuid4

import minari
import numpy as np
import torch
from tqdm import tqdm
import random
import parameters
from torch.nn.functional import mse_loss

from pareto import ParetoSelector
from fqe.fqecriticwrapper import FQECriticWrapper

from data_processor import create_offline_dataset_from_minari
from linear_gp import Program, Mutator

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    offline_data = create_offline_dataset_from_minari(
        minari.load_dataset("HalfCheetah-Expert-v2"),
        shuffle=False
    )

    fqe_critic = FQECriticWrapper()

    population = [ (Program(), uuid4()) for _ in range(parameters.POPULATION_SIZE)]

    timesteps = 0

    for batch in offline_data:
        states, _, rewards, _, next_actions = [
            data.to(device) for data in batch
        ]

        fitnesses = {}
        for individual, individual_id in tqdm(population):
            # Actions are the 6-registers that each individual predicts for a feature set
            actions = [individual.predict(np.array(state.cpu()))[:6] for state in states]
            actions = torch.tensor(actions, device=device)

            # Fitness is the sum Q-value for all state, action pairs.
            fitness = fqe_critic.predict(states, actions).sum().cpu().detach().numpy()

            fitnesses[individual_id] = fitness

        population = sorted(population, key=lambda x: fitnesses[x[1]], reverse=True)[:int(parameters.POP_GAP * parameters.POPULATION_SIZE)]

        parents = random.sample(population, k=int(parameters.POP_GAP * parameters.POPULATION_SIZE))
        for parent, parent_id in parents:
            child = deepcopy(parent)
            Mutator.mutateProgram(child)
            population.append((child, uuid4()))

        median_fitness = np.median(np.array(list(fitnesses.values())))
        best_fitness = np.max(np.array(list(fitnesses.values())))

        print(f"Timestep {timesteps}, Median Fitness: {median_fitness:3}, Best Fitness: {best_fitness:3}")
        timesteps += len(states)

        if timesteps % 10240 == 0:
            with open(f'results/HalfCheetah-v5_{timesteps}.pkl', 'wb') as f:
                pickle.dump({"population": population, "fitnesses": fitnesses}, f)

        if timesteps >= 1_000_000:
            break