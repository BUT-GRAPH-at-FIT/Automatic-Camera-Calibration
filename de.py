import tensorflow as tf
import numpy as np
import sys


def fun(paramsVec):
    return tf.reduce_prod(paramsVec*paramsVec, axis=1)


def funND(paramsVec):
    return tf.reduce_prod(paramsVec*paramsVec, axis=2)


def my_differential_evolution_single(func, bounds, params, iter=1000, max_same_iter=20, popsize=15, mutation=(0.5, 1.0), recombination=0.7, disp=False):
    dimensions = len(bounds)
    while popsize % 3 != 0:
        popsize += 1
    population = tf.convert_to_tensor(np.random.rand(popsize, dimensions), dtype=tf.float64)
    bounds = tf.convert_to_tensor(bounds, dtype=tf.float64)
    min_bounds = tf.minimum(bounds[:,0], bounds[:,1])
    max_bounds = tf.maximum(bounds[:,0], bounds[:,1])
    diff = tf.abs(min_bounds - max_bounds)
    population_denorm = min_bounds + population * diff
    fitness = func(population_denorm, params)

    cost_value = sys.float_info.max
    cost_iter = 0

    for i in range(iter):
        random_trio_1 = tf.reshape(tf.gather(population, tf.random.shuffle(tf.range(popsize))), shape=[-1, 3, dimensions])
        random_trio_2 = tf.reshape(tf.gather(population, tf.random.shuffle(tf.range(popsize))), shape=[-1, 3, dimensions])
        random_trio_3 = tf.reshape(tf.gather(population, tf.random.shuffle(tf.range(popsize))), shape=[-1, 3, dimensions])


        mutation_trios = tf.concat([random_trio_1, random_trio_2, random_trio_3], axis=0)
        vectors_1, vectors_2, vectors_3 = tf.unstack(mutation_trios, axis=1, num=3)
        mutants = vectors_1 + tf.convert_to_tensor(np.random.uniform(mutation[0], mutation[1]), dtype=tf.float64) * (vectors_2 - vectors_3)
        mutants = tf.clip_by_value(mutants, 0, 1)

        crossover_probabilities = tf.convert_to_tensor(np.random.rand(popsize, dimensions), dtype=tf.float64)
        trial_population = tf.where(crossover_probabilities < recombination, x=mutants, y=population)
        trial_denorm = min_bounds + trial_population * diff
        trial_fitness = func(trial_denorm, params)


        cond = tf.tile(tf.expand_dims(trial_fitness < fitness, -1), [1, tf.shape(population)[1].numpy()])
        population = tf.where(cond, x=trial_population, y=population)
        fitness = tf.where(trial_fitness < fitness, x=trial_fitness, y=fitness)

        act_cost = int(round(tf.gather(fitness, tf.argmin(fitness)).numpy()))
        if act_cost < cost_value:
            cost_value = act_cost
            cost_iter = i

        if (i - cost_iter) > max_same_iter:
            break

        if disp:
            print('Iteration {:04d} / {:04d}, error {:.6f}'.format(i+1, iter, tf.gather(fitness, tf.argmin(fitness)).numpy()))#tf.gather(fitness, tf.argmin(fitness)), min_bounds + tf.gather(population, tf.argmin(fitness)) * diff)


    population_denorm = min_bounds + population * diff

    return tf.gather(population_denorm, tf.argmin(fitness))



def my_differential_evolution(func, bounds, params, iter=1000, max_same_iter=20, popsize=15, mutation=(0.5, 1.0), recombination=0.7, disp=False, batch_size=3):
    dimensions = len(bounds)
    while popsize % 3 != 0:
        popsize += 1
    population = tf.convert_to_tensor(np.random.rand(batch_size, popsize, dimensions), dtype=tf.float64)
    bounds = tf.convert_to_tensor(bounds, dtype=tf.float64)
    min_bounds = tf.minimum(bounds[:,0], bounds[:,1])
    max_bounds = tf.maximum(bounds[:,0], bounds[:,1])
    diff = tf.abs(min_bounds - max_bounds)
    population_denorm = min_bounds + population * diff
    fitness = func(population_denorm, params)

    idxs = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size), [-1,1]), [1,popsize]), [batch_size*popsize,1])

    cost_value = sys.float_info.max
    cost_iter = 0

    for i in range(iter):
        idxs_rand_1 = tf.reshape(tf.tile(tf.reshape(tf.random.shuffle(tf.range(popsize)), [1,-1]), [batch_size,1]), [batch_size*popsize,1])
        random_trio_1 = tf.reshape(tf.gather_nd(population, tf.concat((idxs, idxs_rand_1), 1)), [batch_size, -1, 3, dimensions])
        idxs_rand_2 = tf.reshape(tf.tile(tf.reshape(tf.random.shuffle(tf.range(popsize)), [1,-1]), [batch_size,1]), [batch_size*popsize,1])
        random_trio_2 = tf.reshape(tf.gather_nd(population, tf.concat((idxs, idxs_rand_2), 1)), [batch_size, -1, 3, dimensions])
        idxs_rand_3 = tf.reshape(tf.tile(tf.reshape(tf.random.shuffle(tf.range(popsize)), [1,-1]), [batch_size,1]), [batch_size*popsize,1])
        random_trio_3 = tf.reshape(tf.gather_nd(population, tf.concat((idxs, idxs_rand_3), 1)), [batch_size, -1, 3, dimensions])


        mutation_trios = tf.concat([random_trio_1, random_trio_2, random_trio_3], axis=1)
        vectors_1, vectors_2, vectors_3 = tf.unstack(mutation_trios, axis=2, num=3)
        mutants = vectors_1 + tf.convert_to_tensor(np.random.uniform(mutation[0], mutation[1]), dtype=tf.float64) * (vectors_2 - vectors_3)
        mutants = tf.clip_by_value(mutants, 0, 1)

        crossover_probabilities = tf.convert_to_tensor(np.random.rand(batch_size, popsize, dimensions), dtype=tf.float64)
        trial_population = tf.where(crossover_probabilities < recombination, x=mutants, y=population)
        trial_denorm = min_bounds + trial_population * diff
        trial_fitness = func(trial_denorm, params)

        use_trial = tf.tile(tf.expand_dims(trial_fitness < fitness, -1), (1,1,dimensions))

        population = tf.where(use_trial, x=trial_population, y=population)
        fitness = tf.where(trial_fitness < fitness, x=trial_fitness, y=fitness)

        act_cost = int(round(np.mean(tf.gather(fitness, tf.argmin(fitness)).numpy())))
        if act_cost < cost_value:
            cost_value = act_cost
            cost_iter = i

        if (i - cost_iter) > max_same_iter:
            break

        if disp:
            # print(tf.gather(fitness, tf.argmin(fitness)), tf.gather(population, tf.argmin(fitness)))
            print('Iteration inner {:04d} / {:04d}, error {:.6f}'.format(i+1, iter, np.mean(tf.gather(fitness, tf.argmin(fitness)).numpy())))

    population_denorm = min_bounds + population * diff

    return tf.gather_nd(population_denorm, tf.concat((tf.reshape(tf.range(batch_size), [-1,1]), tf.reshape(tf.cast(tf.argmin(fitness, axis=1), tf.int32), [-1,1])), 1))
