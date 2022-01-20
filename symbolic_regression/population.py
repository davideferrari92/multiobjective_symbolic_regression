from symbolic_regression.Program import Program


def generate_population(features,
                        operations,
                        fitness,
                        data,
                        target,
                        weights,
                        const_range,
                        constants_optimization,
                        constants_optimization_conf,
                        parsimony=0.90,
                        parsimony_decay=0.90):
    """
    """
    p = Program(
        features=features,
        operations=operations,
        const_range=const_range,
        constants_optimization=constants_optimization,
        constants_optimization_conf=constants_optimization_conf,
        parsimony=parsimony,
        parsimony_decay=parsimony_decay
    )

    p.init_program()

    p.eval_fitness(fitness=fitness, program=p,
                   data=data, target=target, weights=weights)

    return p
