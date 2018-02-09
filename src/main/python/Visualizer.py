import gensim
import pandas as pd
import smart_open
import random

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from gensim.models import Word2Vec

class Visualizer:

    def __init__(self):
        print("start")

    def saveTensorBoard(self):

        model = Word2Vec.load("YOUR-MODEL")

        # read data
        dataframe = pd.read_csv('movie_plots.csv')
        dataframe


if __name__ == '__main__':
    visualizer = Visualizer()
    visualizer.saveTensorBoard()