from joblib import Parallel, delayed
from symbolic_regression.Program import Program


def generate_population(features,
                        operations,
                        fitness,
                        data,
                        target,
                        weights,
                        const_range,
                        parsimony=0.90,
                        parsimony_decay=0.90):
    """
    """
    p = Program(
        features=features,
        operations=operations,
        const_range=const_range,
        parsimony=parsimony,
        parsimony_decay=parsimony_decay
    )

    p.init_program()

    from symbolic_regression.multiobjective.training import eval_fitness

    p.fitness = eval_fitness(fitness=fitness, program=p,
                             data=data, target=target, weights=weights)

    return p


def generate_population_n(n,
                          features,
                          operations,
                          fitness,
                          data,
                          target,
                          weights,
                          const_range,
                          parsimony,
                          parsimony_decay):
    """
    """

    return Parallel(n_jobs=-1)(
        delayed(generate_population)(
            features, operations, fitness, data, target,
            weights, const_range, parsimony=parsimony,
            parsimony_decay=parsimony_decay
        ) for _ in range(n)
    )
