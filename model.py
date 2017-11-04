import os
import multiprocessing

import numpy as np
import keras
import names
import time
import datetime
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


import accuracy
from constants import *
import keras_model
import tensorflow as tf

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

        self.lrs = np.full([self.epoch_count], 0, dtype='float64')
        self.losses = np.full([self.epoch_count], 0, dtype='float64')
        self.acctrains = np.full([self.epoch_count], 0, dtype='float64')
        self.acctests = np.full([self.epoch_count], 0, dtype='float64')

        for model_name in self.model_names:
            model_info = model_name.split('_')
            time, epoch, lr, loss, acc, trainacc, testacc, lastname = model_info
            epoch = int(epoch)
            lr = float(lr)
            self.lrs[epoch] = float(lr)
            self.losses[epoch] = float(loss)
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

        loss = logs.get('loss', 42)
        acc = logs.get('acc', 42)
        lr = keras.backend.eval(self.m.optimizer.lr)
        epoch = self.epoch_count - 1

        t = datetime.datetime.fromtimestamp(time.mktime(time.localtime()))
        t = time.strftime(time_format)
        lastname = names.get_last_name().lower()
        info = self.eval_accuracies()
        acctrain = info['acctrain']
        acctest = info['acctest']

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

        self.lrs = np.asarray(self.lrs.tolist() + [lr])
        self.losses = np.asarray(self.losses.tolist() + [loss])
        self.acctrains = np.asarray(self.acctrains.tolist() + [acctrain])
        self.acctests = np.asarray(self.acctests.tolist() + [acctest])

        # now = datetime.datetime.now()
        # if self.previous_save is None or (now - self.previous_save).total_seconds() > self.delta_save:
            # self.previous_save = now
            # model_name = self.create_model_name(loss, acc)
        self.m.save(os.path.join(
            self.directory, model_name + '.hdf5'
        ))
        print('-------------------------------------------------- Asking for refresh')
        # self.refresh = True
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
                # keras.callbacks.ReduceLROnPlateau(
                #     'loss', factor=1/2, patience=100, verbose=True, cooldown=5,
                # ),
            ],
        )

    def show_board(self):
        def _show_accuracy(ys, color, label):
            label = '{} (max={})'.format(label, ys.max())
            ax1.plot(ndi.gaussian_filter1d(ys, 3), lw=1, alpha=1, ls='-', c=color, label=label, zorder=3)
            above = np.maximum.accumulate(ys)
            below = np.minimum.accumulate(ys[::-1])[::-1]
            above = ndi.gaussian_filter1d(above, 2)
            below = ndi.gaussian_filter1d(below, 2)
            ax1.fill_between(
                range(above.size), above, below,
                facecolor=color,
                alpha=1/3,
                zorder=0,
            )

        fig, ax1 = plt.subplots(figsize=(16, 9))
        _show_accuracy(self.acctrains, 'red', 'Accuracy train set')
        _show_accuracy(self.acctests, 'green', 'Accuracy test set')
        ax1.plot(self.lrs, lw=2, alpha=1/3, ls='-', c='orange', label='Learning rate', zorder=1)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('percent')
        plt.legend(loc='center right')

        ax2 = ax1.twinx()
        mask = self.losses < self.losses.min() + self.losses.std() * 3
        ax2.plot(
            mask.nonzero()[0],
            self.losses[mask],
            lw=1, alpha=1, ls='-', c='purple',
            label='loss (min={:.8f})'.format(self.losses.min()),
            zorder=2,
        )
        ax2.set_ylabel('loss', color='purple')
        ax2.tick_params('y', colors='purple')
        plt.legend(loc='lower center')

        plt.tight_layout()

        plt.savefig(
            self.directory + '/status.png',
            dpi=180, orientation='landscape', bbox_inches='tight',
        )
