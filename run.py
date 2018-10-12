import gc
import re
import os
import ast
import sys
import random
import threading
import configparser
import librosa
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

###########work with data#############

#read .au
#return songs as melspectrograms
class AudioStruct(object):
  def __init__(self, file_path):
    self.song_samples = 660000
    self.file_path = file_path
    self.n_fft = 2048
    self.hop_length = 512
    self.genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
      'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}

#read .au
#return numpy arrays
  def getdata(self):
    song_data = []
    genre_data = []
    for x,_ in self.genres.items():
      for root, subdirs, files in os.walk(self.file_path + x):
        for file in files:
            file_name = self.file_path + x + "/" + file
            print(file_name)
            signal, sr = librosa.load(file_name)
            melspec = librosa.feature.melspectrogram(signal[:self.song_samples],
              sr = sr, n_fft = self.n_fft, hop_length = self.hop_length).T[:1280,]
            song_data.append(melspec)
            genre_data.append(self.genres[x])
    return np.array(song_data), keras.utils.to_categorical(genre_data, len(self.genres))

############architecture#################

class ModelZoo(object):
  @staticmethod
  def cnn_melspect_1D(input_shape):
    kernel_size = 3
    activation_func = Activation('relu')
    inputs = Input(input_shape)
    # Convolutional block_1
    conv1 = Conv1D(32, kernel_size)(inputs)
    act1 = activation_func(conv1)
    bn1 = BatchNormalization()(act1)
    pool1 = MaxPooling1D(pool_size=2, strides=2)(bn1)
    # Convolutional block_2
    conv2 = Conv1D(64, kernel_size)(pool1)
    act2 = activation_func(conv2)
    bn2 = BatchNormalization()(act2)
    pool2 = MaxPooling1D(pool_size=2, strides=2)(bn2)
    # Convolutional block_3
    conv3 = Conv1D(128, kernel_size)(pool2)
    act3 = activation_func(conv3)
    bn3 = BatchNormalization()(act3)
    # Global Layers
    gmaxpl = GlobalMaxPooling1D()(bn3)
    gmeanpl = GlobalAveragePooling1D()(bn3)
    mergedlayer = concatenate([gmaxpl, gmeanpl], axis=1)
    # Regular MLP
    dense1 = Dense(512,
        kernel_initializer='glorot_normal',
        bias_initializer='glorot_normal')(mergedlayer)
    actmlp = activation_func(dense1)
    reg = Dropout(0.5)(actmlp)
    dense2 = Dense(512,
        kernel_initializer='glorot_normal',
        bias_initializer='glorot_normal')(reg)
    actmlp = activation_func(dense2)
    reg = Dropout(0.5)(actmlp)
    dense2 = Dense(10, activation='softmax')(reg)
    model = Model(inputs=[inputs], outputs=[dense2])
    return model

############Functions for features#########

class AudioUtils(object):
  def __init__(self):
    self.augment_factor = 10

  def random_split(self, x):
    melspec = librosa.feature.melspectrogram(x, n_fft = self.n_fft,
      hop_length = self.hop_length).T
    offset = random.sample(range(0, melspec.shape[0] - 128), 1)[0]
    return melspec[offset:(offset+128)]

  def splitsongs_melspect(self, X, y, cnn_type = '1D'):
    temp_X = []
    temp_y = []
    for i, song in enumerate(X):
      song_slipted = np.split(song, self.augment_factor)
      for s in song_slipted:
        temp_X.append(s)
        temp_y.append(y[i])
    temp_X = np.array(temp_X)
    temp_y = np.array(temp_y)
    if not cnn_type == '1D':
      temp_X = temp_X[:, np.newaxis]
    return temp_X, temp_y

  def voting(self, y_true, pred):
    if y_true.shape[0] != pred.shape[0]:
      raise ValueError('Both arrays should have the same size!')
    arr_size = y_true.shape[0]
    pred = np.split(pred, arr_size/self.augment_factor)
    y_true = np.split(y_true, arr_size/self.augment_factor)
    voting_truth = []
    voting_ans = []
    for x,y in zip(y_true, pred):
      voting_truth.append(mode(x)[0][0])
      voting_ans.append(mode(y)[0][0])
    return np.array(voting_truth), np.array(voting_ans)

class MusicDataGenerator(object):
  def __init__(self,
    time_stretching=False,
    pitch_shifting=False,
    dynamic_range_compression=False,
    background_noise=False):
    self.time_stretching = time_stretching
    self.pitch_shifting = pitch_shifting
    self.background_noise = background_noise
    self.dynamic_range_compression = dynamic_range_compression
    self.audioutils = AudioUtils()

  def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None):
    return NumpyArrayIterator(
      x, y, self,
      batch_size=batch_size,
      shuffle=shuffle,
      seed=seed)

  def random_transform(self, x, seed=None):
    if seed is not None:
      np.random.seed(seed)
    if self.time_stretching:
      factor = np.random.uniform(0.8,1.2)
      x = librosa.effects.time_stretch(x, factor)
    if self.pitch_shifting:
      passif
    if self.background_noise:
      passif
    if self.dynamic_range_compression:
      passif
    x = self.audioutils.random_split(x)
    return x

  def fit(self, x, augment=False, rounds=1, seed=None):
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 2:
      raise ValueError('Input to `.fit()` should have rank 2. '
        'Got array with shape: ' + str(x.shape))
    if seed is not None:
      np.random.seed(seed)
    x = np.copy(x)
    if augment:
      ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
      for r in range(rounds):
        for i in range(x.shape[0]):
          ax[i + r * x.shape[0]] = self.random_transform(x[i])
      x = ax

class Iterator(object):
  def __init__(self, n, batch_size, shuffle, seed):
    self.n = n
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.batch_index = 0
    self.total_batches_seen = 0
    self.lock = threading.Lock()
    self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

  def reset(self):
    self.batch_index = 0

  def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
    self.reset()
    while 1:
      if seed is not None:
        np.random.seed(seed + self.total_batches_seen)
      if self.batch_index == 0:
        index_array = np.arange(n)
        if shuffle:
          index_array = np.random.permutation(n)
      current_index = (self.batch_index * batch_size) % n
      if n > current_index + batch_size:
        current_batch_size = batch_size
        self.batch_index += 1
      else:
        current_batch_size = n - current_index
        self.batch_index = 0
      self.total_batches_seen += 1
      yield (index_array[current_index: current_index + current_batch_size],
        current_index, current_batch_size)

  def __iter__(self):
    return self

  def __next__(self, *args, **kwargs):
    return self.next(*args, **kwargs)

class NumpyArrayIterator(Iterator):
  def __init__(self, x, y, music_data_generator,
    batch_size=32, shuffle=False, seed=None):
    if y is not None and len(x) != len(y):
      raise ValueError('X (music tensor) and y (labels) '
        'should have the same length. '
        'Found: X.shape = %s, y.shape = %s' %
        (np.asarray(x).shape, np.asarray(y).shape))
    self.x = np.asarray(x, dtype=K.floatx())
    if self.x.ndim != 2:
      raise ValueError('Input data in `NumpyArrayIterator` '
        'should have rank 2. You passed an array '
        'with shape', self.x.shape)
    if y is not None:
      self.y = np.asarray(y)
    else:
        self.y = None
    self.music_data_generator = music_data_generator
    super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

  def next(self):
    with self.lock:
      index_array, current_index, current_batch_size = next(self.index_generator)
    melspec_size = [128,128]
    batch_x = np.zeros(tuple([current_batch_size] + melspec_size), dtype=K.floatx())
    for i, j in enumerate(index_array):
      x = self.x[j]
      x = self.music_data_generator.random_transform(x.astype(K.floatx()))
      batch_x[i] = x
    if self.y is None:
      return batch_x
    batch_y = self.y[index_array]
    return batch_x, batch_y

def main():
  config = configparser.ConfigParser()
  config.read('params.ini')
  GTZAN_FOLDER = config['FILE_READ']['GTZAN_FOLDER']
  MODEL_PATH = config['FILE_READ']['SAVE_MODEL']
  SAVE_NPY = ast.literal_eval(config['FILE_READ']['SAVE_NPY'])
  EXEC_TIMES = int(config['PARAMETERS_MODEL']['EXEC_TIMES'])
  CNN_TYPE = config['PARAMETERS_MODEL']['CNN_TYPE']
  batch_size = int(config['PARAMETERS_MODEL']['BATCH_SIZE'])
  epochs = int(config['PARAMETERS_MODEL']['EPOCHS'])
  if not ((CNN_TYPE == '1D') or (CNN_TYPE == '2D')):
    raise ValueError('Argument Invalid: The options are 1D or 2D for CNN_TYPE')
  data_type = config['FILE_READ']['TYPE']
  input_shape = (128, 128)
  print("data_type: %s" % data_type)
  if data_type == 'AUDIO_FILES':
    song_rep = AudioStruct(GTZAN_FOLDER)
    songs, genres = song_rep.getdata()
    if SAVE_NPY:
      np.save(GTZAN_FOLDER + 'songs.npy', songs)
      np.save(GTZAN_FOLDER + 'genres.npy', genres)
  elif data_type == 'NPY':
    songs = np.load(GTZAN_FOLDER + 'songs.npy')
    genres = np.load(GTZAN_FOLDER + 'genres.npy')
  else:
    raise ValueError('Argument Invalid: The options are AUDIO_FILES or NPY for data_type')
  print("Original songs array shape: {0}".format(songs.shape))
  print("Original genre array shape: {0}".format(genres.shape))
#train
  val_acc = []
  test_history = []
  test_acc = []
  test_acc_mvs = []
  for x in range(EXEC_TIMES):
#split dataset
    X_train, X_test, y_train, y_test = train_test_split(
      songs, genres, test_size=0.1, stratify=genres)
    X_train, X_Val, y_train, y_val = train_test_split(
      X_train, y_train, test_size=132, stratify=y_train)
#split data in size 128x128
    X_Val, y_val = AudioUtils().splitsongs_melspect(X_Val, y_val, CNN_TYPE)
    X_test, y_test = AudioUtils().splitsongs_melspect(X_test, y_test, CNN_TYPE)
    X_train, y_train = AudioUtils().splitsongs_melspect(X_train, y_train, CNN_TYPE)
    if CNN_TYPE == '1D':
      cnn = ModelZoo.cnn_melspect_1D(input_shape)
    elif CNN_TYPE == '2D':
      cnn = ModelZoo.cnn_melspect_2D(input_shape)
    print("\nTrain shape: {0}".format(X_train.shape))
    print("Validation shape: {0}".format(X_Val.shape))
    print("Test shape: {0}\n".format(X_test.shape))
    print("Size of the CNN: %s\n" % cnn.count_params())
#optimizers
    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-5, nesterov=True)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
    cnn.compile(loss=keras.losses.categorical_crossentropy,
      optimizer=sgd,
      metrics=['accuracy'])
#early stop
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',
      min_delta=0,
      patience=2,
      verbose=0,
      mode='auto')
#fit the model
    history = cnn.fit(X_train, y_train,
      batch_size=batch_size,
      epochs=epochs,
      verbose=1,
      validation_data=(X_Val, y_val),
      callbacks = [earlystop])
    score = cnn.evaluate(X_test, y_test, verbose=0)
    score_val = cnn.evaluate(X_Val, y_val, verbose=0)

#Majority Voting System
    pred_values = np.argmax(cnn.predict(X_test), axis = 1)
    mvs_truth, mvs_res = AudioUtils().voting(np.argmax(y_test, axis = 1), pred_values)
    acc_mvs = accuracy_score(mvs_truth, mvs_res)
    val_acc.append(score_val[1])
    test_acc.append(score[1])
    test_history.append(history)
    test_acc_mvs.append(acc_mvs)
    print('Test accuracy:', score[1])
    print('Test accuracy for Majority Voting System:', acc_mvs)
    cm = confusion_matrix(mvs_truth, mvs_res)
    print(cm)
  print("Validation accuracy - mean: %s, std: %s" % (np.mean(val_acc), np.std(val_acc)))
  print("Test accuracy - mean: %s, std: %s" % (np.mean(test_acc), np.std(test_acc)))
  print("Test accuracy MVS - mean: %s, std: %s" % (np.mean(test_acc_mvs), np.std(test_acc_mvs)))
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
#save the model
  cnn.save(MODEL_PATH)
#free memory
  del songs
  del genres
  gc.collect()

if __name__ == '__main__':
  main()
