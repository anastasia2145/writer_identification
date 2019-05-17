import sys 
sys.path.insert(0, '../')
import numpy as np
import pandas as pd
from random import  sample
from sklearn.utils import shuffle
from itertools import combinations

from tensorflow.python.keras.utils.data_utils import Sequence
from utils.preprocessing import fetch, resize, pad

     
class WordsSequence(Sequence):
    def __init__(self, img_dir, input_shape, batch_size, x_set, y_set=None):
        if y_set is not None:
            self.x, self.y = x_set, y_set
            self.x, self.y = shuffle(self.x, self.y)
            self.dataset = pd.DataFrame(data={'x': self.x, 'y': self.y, 'used': np.zeros_like(self.y)})
        else:
            self.x, self.y = x_set, None

        self.img_dir = img_dir
        self.input_shape = input_shape
        self.batch_size = batch_size


    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        if self.y is None:
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            return np.array([self.preprocess(fetch(self.img_dir, name)) for name in batch_x])
        
        unused = self.dataset.loc[self.dataset['used'] == 0]
        if len(unused) >= self.batch_size:
            batch_indices = unused.sample(n=self.batch_size).index
        else:
            batch_indices = unused.sample(n=self.batch_size, replace=True).index

        self.dataset.loc[batch_indices, 'used'] = 1
        batch_x = self.dataset.iloc[batch_indices]['x'].values
        batch_y = self.dataset.iloc[batch_indices]['y'].values     
        
        pairs, labels = self.AllPositivePairSelector(batch_x, batch_y)
      
        anchor_images = pairs[:,0]
        positiv_images = pairs[:,1]
        return [np.array([self.preprocess(fetch(self.img_dir, img)) for img in anchor_images]), 
            np.array([self.preprocess(fetch(self.img_dir, img)) for img in positiv_images]), labels], []

    def preprocess(self, img):
        assert len(img.shape) == 3

        h, w, _ = img.shape
        if h / w <= self.input_shape[0] / self.input_shape[1]:
            img = resize(img, (self.input_shape[1], int(self.input_shape[1] * h / w)))
        else:
            img = resize(img, (int(self.input_shape[0] * w / h), self.input_shape[0]))

        img = pad(img, (self.input_shape[1], self.input_shape[0]))
        return img / 255.  # pixel normalization
    
    def AllPositivePairSelector(self, x, y):
        all_ind_pairs = np.array(list(combinations(range(len(y)), 2)))

        positive_inds = all_ind_pairs[y[all_ind_pairs[:,0]] == y[all_ind_pairs[:,1]]]
        
        negative_inds = all_ind_pairs[y[all_ind_pairs[:,0]] != y[all_ind_pairs[:,1]]]
       
        positive_pairs = x[positive_inds]
        positive_labels = [1 for _ in range(len(positive_pairs))]
        
        negative_pairs = x[negative_inds]
        if len(positive_pairs) == 0:
            pair = negative_pairs[0]
            return np.array([pair]), np.array([0])
            
        shuffle_inds = sample([i for i in range(len(negative_pairs))], len(positive_pairs))
        negative_pairs = negative_pairs[shuffle_inds]
        negative_labels = [0 for _ in range(len(negative_pairs))]
        
        lables = positive_labels + negative_labels
        pairs = np.concatenate((positive_pairs, negative_pairs), axis = 0)
        
        return pairs, np.array(lables)
        
    def on_epoch_end(self):        
        if self.y is not None:
            self.dataset = pd.DataFrame(data={'x': self.x, 'y': self.y, 'used': np.zeros_like(self.y)})
            self.dataset = self.dataset.sample(n=len(self.dataset))
