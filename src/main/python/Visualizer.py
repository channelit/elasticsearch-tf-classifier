from _config import ConfigMap
import sys, os
import gensim
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

tb = ConfigMap("TensorBoard")
training = ConfigMap("Training")


class Visualizer:

    def __init__(self):
        self.doc_model_file = os.path.join(training['basedir'], 'doc_model')
        self.doc_model = gensim.models.Word2Vec.load(self.doc_model_file)
        self.word_model_file = os.path.join(training['basedir'], 'word2vec_model_all')
        self.word_model = gensim.models.Word2Vec.load(self.word_model_file)
        self.log_folder = os.path.join(tb['basedir'])

    def visualize_doc_model(self):
        meta_file = "doc_w2x_metadata.tsv"
        placeholder = np.zeros((len(self.doc_model.wv.index2word), 200))

        with open(os.path.join(self.log_folder, meta_file), 'wb') as file_metadata:
            for i, word in enumerate(self.doc_model.wv.index2word):
                placeholder[i] = self.doc_model[word]
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

        sess = tf.InteractiveSession()

        embedding = tf.Variable(placeholder, trainable=False, name='doc_w2x_metadata')
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self.log_folder, sess.graph)

        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'doc_w2x_metadata'
        embed.metadata_path = meta_file

        projector.visualize_embeddings(writer, config)
        saver.save(sess, os.path.join(self.log_folder, 'doc_w2x_metadata.ckpt'))
        print('done')

    def visualize_word_model(self):
        meta_file = "word_w2x_metadata.tsv"
        placeholder = np.zeros((len(self.word_model.wv.index2word), 100))

        with open(os.path.join(self.log_folder, meta_file), 'wb') as file_metadata:
            for i, word in enumerate(self.word_model.wv.index2word):
                placeholder[i] = self.word_model[word]
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

        sess = tf.InteractiveSession()

        embedding = tf.Variable(placeholder, trainable=False, name='word_w2x_metadata')
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self.log_folder, sess.graph)

        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'word_w2x_metadata'
        embed.metadata_path = meta_file

        projector.visualize_embeddings(writer, config)
        saver.save(sess, os.path.join(self.log_folder, 'word_w2x_metadata.ckpt'))
        print('done')


if __name__ == "__main__":
    visualizer = Visualizer()
    # visualizer.visualize_doc_model()
    visualizer.visualize_word_model()
