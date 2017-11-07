import os
import multiprocessing
import time
import datetime
import functools

import numpy as np
import names
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import pandas as pd

import accuracy
from constants import *
import model_show

class Model(model_show.ModelShow):
    """
    model name: time_epoch_lr_loss_acc_trainacc_testacc_lastname
    """

    def __init__(self, directory, ds):
        print('Constructing Model class')
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        self.ds = ds
        self.directory = directory
        self.model_names = self.create_model_names()
        self.previous_save = None


        self.df = pd.DataFrame()
        for model_name in self.model_names:
            model_info = model_name.split('_')
            time, epoch, lr, loss, acc, acctrain, acctest, lastname = model_info

            self.df = self.df.append(pd.Series({
                'epoch': int(epoch),
                'lr': float(lr),
                'loss': float(loss),
                'acctrain': float(acctrain),
                'acctest': float(acctest),
            }), ignore_index=True)
        if self.df.size > 0:
            self.df.loc[:, 'epoch'] = pd.to_numeric(self.df.epoch, downcast='integer')

    @property
    @functools.lru_cache(1)
    def keras(self):
        import keras
        return keras

    @property
    @functools.lru_cache(1)
    def m(self):
        import keras_model

        if len(self.model_names) == 0:
            print('Creating keras model from scratch...')
            return keras_model.create_model()
        else:
            path = os.path.join(self.directory, self.model_names[-1] + '.hdf5')
            print('Creating keras model from file {}'.format(
                path
            ))
            return self.keras.models.load_model(path)

    def create_model_names(self):
        return sorted([fname[:-5]
             for fname in os.listdir(self.directory)
             if fname.endswith('.hdf5')
        ])

    def on_epoch_end(self, epoch, logs=None):
        print()
        if self.df.size == 0:
            epoch = 0
        else:
            epoch = int(self.df.epoch.max() + 1)

        loss = logs.get('loss', 42)
        acc = logs.get('acc', 42)
        lr = self.keras.backend.eval(self.m.optimizer.lr)

        t = datetime.datetime.fromtimestamp(time.mktime(time.localtime()))
        t = time.strftime(time_format)
        lastname = names.get_last_name().lower()
        info = self.eval_accuracies()
        acctrain = info['acctrain']
        acctest = info['acctest']

        self.df = self.df.append(pd.Series({
            'epoch': epoch,
            'lr': lr,
            'loss': loss,
            'acctrain': acctrain,
            'acctest': acctest,
        }), ignore_index=True)


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
        model_name = fmt.format(t, epoch, lr, loss, acc, acctrain, acctest, lastname)
        # now = datetime.datetime.now()
        # if self.previous_save is None or (now - self.previous_save).total_seconds() > self.delta_save:
            # self.previous_save = now
            # model_name = self.create_model_name(loss, acc)
        self.m.save(os.path.join(
            self.directory, model_name + '.hdf5'
        ))
        self.show_board()

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

    def fit(self, epoch_count=1000, delta_save=0):

        self.delta_save = delta_save
        self.previous_save = None

        self.m.fit(
            self.ds.xtrain, self.ds.ytrain,
            epochs=epoch_count,
            batch_size=1,
            verbose=1,
            callbacks=[
                self.keras.callbacks.LambdaCallback(on_epoch_end=self.on_epoch_end),
                # keras.callbacks.ReduceLROnPlateau(
                #     'loss', factor=1/2, patience=100, verbose=True, cooldown=5,
                # ),
            ],
        )
