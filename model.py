
import os

import numpy as np
import keras
import names
import time
# import pytz
import datetime

import accuracy
import paths
from constants import *
import keras_model

class Model(object):
    """
    model name: time_epoch_lr_loss_acc_trainacc_testacc_lastname
    """

    def __init__(self, directory, ds):
        print('Constructing Model')
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        self.ds = ds
        self.directory = directory
        self.epoch_count_path = os.path.join(directory, 'epoch_count')
        self.model_names = self.create_model_names()
        self.previous_save = None

        if len(self.model_names) == 0:
            print('Creating keras model from scratch...')
            self.m = keras_model.create_model()
            self.epoch_count = 0
            open(self.epoch_count_path, 'w')
            assert self.read_epoch_count() == 0
        else:
            path = os.path.join(self.directory, self.model_names[-1] + '.hdf5')
            print('Creating keras model from file {}'.format(
                path
            ))
            self.m = keras.models.load_model(path)
            self.epoch_count = self.read_epoch_count()
            print('  {} epochs'.format(self.epoch_count))

        self.losses = np.full([self.epoch_count], np.nan)
        self.accs = np.full([self.epoch_count], np.nan)
        self.acctrains = np.full([self.epoch_count], np.nan)
        self.acctests = np.full([self.epoch_count], np.nan)

        for model_name in self.model_names:
            model_info = model_name.split('_')
            time, epoch, lr, loss, acc, trainacc, testacc, lastname = model_info
            epoch = int(epoch)
            lr = float(lr)
            self.losses[epoch] = float(loss)
            self.accs[epoch] = float(acc)
            self.acctrains[epoch] = float(trainacc)
            self.acctests[epoch] = float(testacc)

    def read_epoch_count(self):
        return len(open(self.epoch_count_path, 'r').read())

    def create_model_names(self):
        return sorted([fname[:-5]
             for fname in os.listdir(self.directory)
             if fname.endswith('.hdf5')
        ])

    def on_epoch_end(self, epoch, logs=None):
        print()
        open(self.epoch_count_path, 'a').write('x')
        self.epoch_count += 1
        self.losses = np.pad(self.losses, 1, 'constant', constant_values=np.nan)
        self.accs = np.pad(self.accs, 1, 'constant', constant_values=np.nan)
        self.acctrains = np.pad(self.acctrains, 1, 'constant', constant_values=np.nan)
        self.acctests = np.pad(self.acctests, 1, 'constant', constant_values=np.nan)
        # Todo add real values in arrays

        loss = logs.get('loss', 42)
        acc = logs.get('acc', 42)

        now = datetime.datetime.now()
        if self.previous_save is None or (now - self.previous_save).total_seconds() > self.delta_save:
            self.previous_save = now
            model_name = self.create_model_name(loss, acc)
            self.m.save(os.path.join(
                self.directory, model_name + '.hdf5'
            ))

    def create_model_name(self, loss, acc):
        # time = pytz.utc.localize(datetime.datetime.now())
        # time = time.astimezone(pytz.timezone('Europe/Paris'))
        t = datetime.datetime.fromtimestamp(time.mktime(time.localtime()))
        t = time.strftime(time_format)
        epoch = self.epoch_count - 1
        lr = keras.backend.eval(self.m.optimizer.lr)

        info = self.eval_accuracies()
        trainacc = info['acctrain']
        testacc = info['acctest']
        lastname = names.get_last_name().lower()

        fmt = '_'.join([
            '{}', # time
            '{:04d}', # epoch
            '{:010.8f}', # lr
            '{:010.8f}', # loss
            '{:010.8f}', # acc
            '{:05.3f}', # acctrain
            '{:04.2f}', # acctest
            '{}', # lastname
        ])
        return fmt.format(
            t, epoch, lr, loss, acc, trainacc, testacc, lastname
        )

    def eval_accuracies(self):
        print('Predicting training set...')
        ptrain = self.m.predict(self.ds.xtrain, 1, False)
        print('Predicting test set...')
        ptest = self.m.predict(self.ds.xtest, 1, False)

        print('Watching train heat maps...')
        acctrain = accuracy.accuracy(self.ds.ytrain, ptrain)
        print('Watching test heat maps...')
        acctest = accuracy.accuracy(self.ds.ytest, ptest)
        return {
            'acctrain': acctrain,
            'acctest': acctest,
        }

    def fit(self, epoch_count=10000, delta_save=0):

        self.delta_save = delta_save
        self.previous_save = None
        self.m.fit(
            self.ds.xtrain, self.ds.ytrain,
            epochs=epoch_count,
            batch_size=1,
            verbose=1,
            callbacks=[
                keras.callbacks.LambdaCallback(on_epoch_end=self.on_epoch_end),
                keras.callbacks.ReduceLROnPlateau(
                    'loss', patience=10, verbose=True, cooldown=3,
                ),
            ],
        )
