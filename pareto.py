import parameters


class ParetoSelector:
    @staticmethod
    def is_dominated(fitness1, fitness2):
        """Check if fitness1 is dominated by fitness2."""
        return (fitness2[0] >= fitness1[0] and fitness2[1] > fitness1[1]) or (fitness2[0] > fitness1[0] and fitness2[1] >= fitness1[1])

    @staticmethod
    def dominance_count_ranking(population, fitnesses):
        """
        Calculate the dominance count for each individual in the population.

        :param population: List of (individual, individual_id) tuples
        :param fitnesses: Dictionary mapping individual IDs to fitness tuples
        :return: List of tuples (individual, individual_id, dominance_count), sorted by dominance count in descending order
        """
        dominance_counts = {}
        # Initialize dominance counts
        for _, ind_id in population:
            dominance_counts[ind_id] = 0

        # Calculate how many individuals each one dominates
        for _, ind_id in population:
            for _, other_id in population:
                if ind_id != other_id and ParetoSelector.is_dominated(fitnesses[other_id], fitnesses[ind_id]):
                    dominance_counts[ind_id] += 1

        # Create a list of individuals with their dominance count
        ranked_population = [(individual, ind_id, dominance_counts[ind_id]) for individual, ind_id in population]

        # Sort individuals by dominance count (higher is better)
        ranked_population.sort(key=lambda x: x[2], reverse=True)

        return ranked_population

    @staticmethod
    def select_popgap_individuals(population, fitnesses):
        num_to_select = int(parameters.POP_GAP * parameters.POPULATION_SIZE)

        # Get dominance count rankings
        ranked_population = ParetoSelector.dominance_count_ranking(population, fitnesses)

        # Select the top individuals based on the calculated dominance count
        selected_population = ranked_population[:num_to_select]

        # Return only the (individual, individual_id) tuples
        return [(individual, ind_id) for individual, ind_id, _ in selected_population]