import numpy as np
from keras.utils import Sequence
from keras.models import model_from_json

def dataGenerator(n, learning_frames=2, steps=1, repeat=False):
    training = []
    testing = []

    last = 0
    for i in range(learning_frames,n, learning_frames):
        if i+steps < n:
            training.append(np.arange(last,i))
            testing.append(i+steps)
            last = i
    return np.asarray(training), np.asarray(testing)

class Generator(Sequence):

    def __init__(self, x_set, y_set, data, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.data = data

    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.data[self.x[idx * self.batch_size:(idx + 1) * self.batch_size]]
        batch_y = self.data[self.y[idx * self.batch_size:(idx + 1) * self.batch_size]]
        return batch_x, batch_y



class Generator2D(Sequence):

    def __init__(self, x_set, y_set, data1,data2, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.data1[self.x[idx * self.batch_size:(idx + 1) * self.batch_size]]
        batch_y = np.log(1+self.data1[self.y[idx * self.batch_size:(idx + 1) * self.batch_size]])

        batch2_x = self.data2[self.x[idx * self.batch_size:(idx + 1) * self.batch_size]]
        return [batch_x, batch2_x], batch_y


def load_model_json(pmodel, pweight):
    json_file = open(pmodel, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(pweight)
    return model