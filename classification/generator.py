import sys 
sys.path.insert(0, '../')
import numpy as np
import pandas as pd
from keras.utils import Sequence
from sklearn.utils import shuffle
from utils.preprocessing import fetch, resize, pad

from keras.utils import to_categorical

    
class WordsSequence(Sequence):
    def __init__(self, img_dir, input_shape, x_set, y_set=None, batch_size=1, classification=False):
        if classification:
            if y_set is not None:
                self.x, self.y = x_set, y_set
                self.dataset = pd.DataFrame(data={'x': self.x, 'y': self.y, 'used': np.zeros_like(self.y)})
            else:
                self.x, self.y = x_set, None
        else:
            if y_set is not None:
                self.x, self.y = x_set, y_set
                self.x, self.y = shuffle(self.x, self.y)
            else:
                self.x, self.y = x_set, None

        self.img_dir = img_dir
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.classification = classification


    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        if self.classification:
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
            return np.array([self.preprocess(fetch(self.img_dir, name)) for name in batch_x]), to_categorical(batch_y, 95)

        if self.y is None:
            x = self.x[idx]
            return np.expand_dims(self.preprocess(fetch(self.img_dir, x)), axis=0)
            
        
        curr_x = self.x[idx]
        curr_y = self.y[idx]

        x_1_images = self.preprocess(fetch(self.img_dir, curr_x[0]))
        x_2_images = self.preprocess(fetch(self.img_dir, curr_x[1]))
        return [np.expand_dims(x_1_images, axis=0), np.expand_dims(x_2_images, axis=0)], np.array([curr_y])
       

    def preprocess(self, img):
        assert len(img.shape) == 3

        h, w, _ = img.shape
        if h / w <= self.input_shape[0] / self.input_shape[1]:
            img = resize(img, (self.input_shape[1], int(self.input_shape[1] * h / w)))
        else:
            img = resize(img, (int(self.input_shape[0] * w / h), self.input_shape[0]))

        img = pad(img, (self.input_shape[1], self.input_shape[0]))
        return img / 255. 

    def on_epoch_end(self):
        if (not self.classification) and (self.y is not None):
            self.x, self.y = shuffle(self.x, self.y)
        
        if self.classification and self.y is not None:
            self.dataset = pd.DataFrame(data={'x': self.x, 'y': self.y, 'used': np.zeros_like(self.y)})
            self.dataset = self.dataset.sample(n=len(self.dataset))
