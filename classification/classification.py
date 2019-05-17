import os
import numpy as np
import pandas as pd
from collections import Counter
from generator import WordsSequence


from keras import backend as K
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.applications.mobilenet import MobileNet
from keras.layers import Input, Dense, Lambda, Dropout, Activation
from keras.optimizers import rmsprop, Adam, SGD, Adagrad


from keras.utils import to_categorical
from keras.models import Model


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
    
    
class Classification_model:
    def __init__(self, alpha, input_shape, num_classes, cache_dir, train_head, batch_size=32):
        self.alpha = alpha
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.build_model(train_head)
    
    def build_model(self, train_head):        
        # If imagenet weights are being loaded, alpha can be one of`0.25`, `0.50`, `0.75` or `1.0` only.
        base_model = MobileNet(input_shape=self.input_shape, alpha=self.alpha, weights='imagenet', include_top=False, pooling='avg') 
        
        # Use to pretrain head
        # if train_head:
            # for layer in base_model.layers[:-4]:
                # layer.trainable = False
            
        # base_model.summary()
        op = Dense(128, activation='relu')(base_model.output)
        op = Dropout(0.00001)(op)
        output_tensor = Dense(self.num_classes, activation='softmax')(op)
       
        self.model = Model(inputs=base_model.input, outputs=output_tensor)
        self.model.summary()
        
    def train(self, train_dir, train_csv, epochs, learning_rate=0.00001):
        train = pd.read_csv(train_csv)
        train_x, train_y = train['file_name'].as_matrix(), train['label'].as_matrix()
        
        self.str2ind_dict, self.ind2str_dict = get_str2numb_numb2dict(train_y)
        train_y = np.array(apply_dict(self.str2ind_dict, train_y))

        train_generator = WordsSequence(img_dir=train_dir,
                                        input_shape = self.input_shape,
                                        x_set=train_x,
                                        y_set=train_y,
                                        batch_size=self.batch_size,
                                        classification=True)
                                        
        optimize = rmsprop(lr=learning_rate, decay=1e-6)
        # optimize = Adam(lr=0.00000001) 
        # optimize = SGD()
        # optimize = Adagrad(lr=0.0001)
        
        self.model.compile(loss='categorical_crossentropy', optimizer=optimize, metrics=['categorical_accuracy'])
        self.model.fit_generator(train_generator,
                                 steps_per_epoch=len(train_x)//self.batch_size,
                                 shuffle=True,
                                 epochs=epochs,
                                 verbose=1, 
                                 callbacks=[ModelCheckpoint(filepath=os.path.join(self.cache_dir, 'checkpoint-{epoch:02d}.h5'), save_weights_only=True)])
        
        self.model.save(os.path.join(self.cache_dir, 'final_model.h5'))
        self.save_weights(os.path.join(self.cache_dir, 'final_weights.h5'))  
        
    def save_weights(self, filename):
        self.model.save_weights(filename)
        
    def load_weights(self, filename):
        self.model.load_weights(filename, by_name=True, skip_mismatch=True)
        
    def predict(self, test_dir, test_csv): 

        test = pd.read_csv(test_csv)
        test_x, test_y = test['file_name'].as_matrix(), test['label'].as_matrix()
        self.str2ind_dict, self.ind2str_dict = get_str2numb_numb2dict(test_y)
        test_generator = WordsSequence(img_dir=test_dir,
                                        input_shape = self.input_shape,
                                        x_set=test_x,
                                        batch_size=self.batch_size,
                                        classification=True)
                                        
        pred = np.argmax(self.model.predict_generator(test_generator, verbose=1), axis=1)  
        res = np.array(apply_dict(self.ind2str_dict, pred))
        
        count = 0
        for i,j in zip(res, test_y):
            if i == j:
                count += 1
        print('word accuracy: ', count / len(test_y))
        
        count = 0
        autors = np.unique(test_y)
        autor_ind = [np.argwhere(test_y == a) for a in autors]
        for i,inds in enumerate(autor_ind):
            p = Counter(np.ravel(res[inds])).most_common(1)[0][0]
            if p == autors[i]:
                count += 1

        print('top-1 autor accuracy: ', count / len(autors))
        
        сount = 0
        for i,inds in enumerate(autor_ind):
            p = [pair[0] for pair in Counter(np.ravel(res[inds])).most_common(5)]
            if autors[i] in p:
                сount += 1

        print('top-5 autor accuracy: ', сount / len(autors))
