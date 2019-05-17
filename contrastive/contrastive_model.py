import os
import random
import numpy as np
import pandas as pd
from random import  sample
from collections import Counter
from itertools import combinations


import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.applications.mobilenet import MobileNet

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.python.keras.utils.generic_utils import Progbar

from sklearn.neighbors import KNeighborsClassifier
from contrastive_generator import WordsSequence

def get_str2numb_numb2dict(vect):
    str_to_ind_dict = {}
    count = 0
    for v in vect:
        if v not in str_to_ind_dict.keys():
            str_to_ind_dict[v] = count
            count += 1
    reverse_dict = {v:k for k, v in str_to_ind_dict.items()}
    return str_to_ind_dict, reverse_dict

def apply_dict(dict_keys, X):
    res = []
    for x in X:
        res.append(dict_keys[x])
    return res
    

@tf.function
def contrastive_loss(y_true, embeddings_anchor, embeddings_positive, margin=1.0):
    distances = tf.math.sqrt(
        tf.math.reduce_sum(
            tf.math.squared_difference(
                embeddings_anchor, embeddings_positive),
            1))

    return tf.math.reduce_mean(
        tf.cast(y_true, tf.dtypes.float32) * tf.math.square(distances) +
        (1. - tf.cast(y_true, tf.dtypes.float32)) *
        tf.math.square(tf.math.maximum(margin - distances, 0.)),
        name='contrastive_loss')



class ContrastiveLossLayer(Layer):
    def __init__(self, margin=1.0, name=None):
        super(ContrastiveLossLayer, self).__init__(name=name)
        self._margin = margin

    def __call__(self, y_true, embeddings_anchor, embeddings_positive):
        return super(ContrastiveLossLayer, self).__call__([y_true, embeddings_anchor, embeddings_positive])

    def call(self, inputs):
        loss = contrastive_loss(*inputs, margin=self._margin)
        self.add_loss(loss)
        return loss


class ProgbarLossLogger(Callback):
    def __init__(self):
        super(ProgbarLossLogger, self).__init__()

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.target = self.params['steps']

        if self.epochs > 1:
            print('Epoch %d/%d' % (epoch + 1, self.epochs))
        self.progbar = Progbar(target=self.target, verbose=True, stateful_metrics=['loss'])

    def on_batch_begin(self, batch, logs=None):
        if self.seen < self.target:
            self.log_values = []

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        num_steps = logs.get('num_steps', 1)
        self.seen += num_steps

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        self.progbar.update(self.seen, self.log_values)
        
        
class ContrastiveModel:
    def __init__(self, alpha, input_shape, cache_dir, batch_size=32):
        self.alpha = alpha
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.embeddings = None
        self.cache_dir = cache_dir
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.base_model = self.build_base_model()
        self.model = self.build_model()
        self.model.summary()
        
        
        
    def build_base_model(self):
        base_network = MobileNet(input_shape=self.input_shape, alpha=self.alpha, weights=None, include_top=False, pooling='avg')        
        op = Dense(128, activation='relu')(base_network.output)
        output = Lambda(lambda x: K.l2_normalize(x, axis=1))(op)       
        return Model(inputs=base_network.input, outputs=output)
        
    def build_model(self):
        # self.base_model.load_weights('classification_cache/checkpoint-02.h5', by_name=True)
        input_a = Input(shape=self.input_shape, name="input_a")
        input_b = Input(shape=self.input_shape, name="input_b")
        labels = Input(shape=(1,), name="labels")
        
        output_a = self.base_model(input_a)
        output_b = self.base_model(input_b)

        outputs = ContrastiveLossLayer()(labels, output_a, output_b)
        model = Model([input_a, input_b, labels], outputs=outputs)
        return model
    
        
    def train(self, train_dir, train_csv, validation_dir, validation_csv, epochs, learning_rate=0.001, margin=1):
        train = pd.read_csv(train_csv)
        validation = pd.read_csv(validation_csv)
        x_train, y_train = train['file_name'].as_matrix(), train['label'].as_matrix()
        x_validation, y_validation = validation['file_name'].as_matrix(), validation['label'].as_matrix()
        
        str2ind_train_dict, ind2str_train_dict = get_str2numb_numb2dict(y_train)
        y_train = np.array(apply_dict(str2ind_train_dict, y_train))

        str2ind_val_dict, ind2str_val_dict = get_str2numb_numb2dict(y_validation)
        y_validation = np.array(apply_dict(str2ind_val_dict, y_validation))
        
        
        train_generator = WordsSequence(train_dir, input_shape=self.input_shape, x_set=x_train, y_set=y_train, batch_size=self.batch_size)
        # validation_generator = WordsSequence(validation_dir, input_shape=self.input_shape, x_set=validation_pairs, y_set=validation_y, batch_size=batch_size)        

        optimize = Adam(lr=0.00001)
        self.model.summary()
        self.model.compile(optimizer=optimize)
        
        self.model.fit_generator(train_generator, shuffle=True, epochs=epochs, verbose=1, 
        callbacks=[ModelCheckpoint(filepath=os.path.join(self.cache_dir, 'checkpoint-{epoch:02d}.h5'), save_weights_only=True)])
        
        self.model.save('final_model.h5')
        self.save_weights('final_weights.h5')
    
    def save_embeddings(self, filename):
        self.embeddings.to_pickle(filename)
    
    def load_embeddings(self, filename):
        self.embeddings = pd.read_pickle(filename)
        
    def save_weights(self, filename):
        self.model.save_weights(filename)
        
    def load_weights(self, filename):
        self.model.load_weights(filename, by_name=True) 
        
    def make_embeddings(self, img_dir, csv, batch_size=1):
        if self.embeddings is not None:
            self.clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
            self.clf.fit(self.embeddings[0][0], self.embeddings[0][1])
        else:
            data = pd.read_csv(csv)
            x, y = data['file_name'].as_matrix(), data['label'].as_matrix()
            
            self.str2ind_test_dict, self.ind2str_test_dict = get_str2numb_numb2dict(y)
            y = np.array(apply_dict(self.str2ind_test_dict, y))

            words = WordsSequence(img_dir, input_shape=self.input_shape, x_set=x, batch_size=batch_size)
            pred = self.base_model.predict_generator(words, verbose=1)

            self.clf = KNeighborsClassifier(n_neighbors=20, metric='euclidean')
            self.clf.fit(pred, y) 
     
            # self.embeddings =  pd.DataFrame(data=[pred, y])
            # self.save_embeddings('embeddings_contrastive.pkl')
    
    def predict(self, img_dir, test_csv, batch_size=1):
        test = pd.read_csv(test_csv)
        x_test, y_test = test['file_name'].as_matrix(), test['label'].as_matrix()
        
        str2ind_test_dict, ind2str_test_dict = get_str2numb_numb2dict(y_test)
        # test_y = np.array(apply_dict(str2ind_test_dict, y_test))

        words = WordsSequence(img_dir, input_shape=self.input_shape, x_set=x_test, batch_size=batch_size)
        test_embeddings = self.base_model.predict_generator(words, verbose=1)

        res = self.clf.predict(test_embeddings) 
     
        predict = np.array(apply_dict(ind2str_test_dict, res))
        count = 0
        for i,j in zip(predict, y_test):
            if i == j:
                count += 1

        print('word accuracy: ', count / len(y_test))
        
        count = 0
        autors = np.unique(y_test)
        autor_ind = [np.argwhere(y_test == a) for a in autors]
        
        for i,inds in enumerate(autor_ind):
            p = Counter(np.ravel(predict[inds])).most_common(1)[0][0]
            if p == autors[i]:
                count += 1

        print('accuracy: ', count / len(autors))
        
        count = 0
        for i,inds in enumerate(autor_ind):
            p = [pair[0] for pair in Counter(np.ravel(predict[inds])).most_common(5)]
            if autors[i] in p:
                count += 1

        print('top-5 autor accuracy: ', count / len(autors))
