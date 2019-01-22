import random

from deap import creator, base, tools, algorithms
import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import load_model
from keras import backend as K


class Model:
    def __init__(self):
        self._pretrained_model_path = 'model-fashion-mnist.h5'
        self._is_fashion_mnist = True
        self._model = self._load_model()
        self._data = self._load_data()

    def run(self):
        self._run_evolution()

    def _load_model(self):
        return load_model(self._pretrained_model_path)

    def _load_data(self):
        num_classes = 10
        img_rows, img_cols = 28, 28

        if self._is_fashion_mnist:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

        else:
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        # reshape data
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        if K.image_data_format() == 'channels_first':
            input_shape = (1, img_rows, img_cols)
        else:
            input_shape = (img_rows, img_cols, 1)
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

    def _eval_evolution_by_loss(self, individual):
        return self._eval_evolution(individual, 0)

    def _eval_evolution_by_acc(self, individual):
        return self._eval_evolution(individual, 1)

    def _eval_evolution(self, individual, score_idx):
        individual_model = self._individual2model(individual)
        scores = individual_model.evaluate(self._data['x_train'], self._data['_y_train'], verbose=0)
        # NOTE: Maybe we want to improve on validation and not train.
        return scores[score_idx]

    def _individual2model(self, individual):
        # TODO: Create a model with neurons working according to the neurons map in `individual` (list of logicals)
        layer_shapes = []
        layer_num_neurons = []
        for layer in self._model.layers:
            shape = layer.output_shape[1:]
            layer_shapes.append(shape)
            layer_num_neurons.append(np.prod(shape))
        assert len(individual) == np.sum(layer_num_neurons)

    def _run_evolution(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self._eval_evolution_by_loss)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=300)

        NGEN = 40
        for gen in range(NGEN):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))
        top10 = tools.selBest(population, k=10)
        return top10


def main():
    model = Model()
    model.run()


if __name__ == '__main__':
    main()
