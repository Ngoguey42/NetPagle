import os
import multiprocessing

import numpy as np
import keras
import names
import time
import datetime
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import pandas as pd

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
        self.model_names = self.create_model_names()
        self.previous_save = None

        if len(self.model_names) == 0:
            print('Creating keras model from scratch...')
            self.m = keras_model.create_model()
        else:
            path = os.path.join(self.directory, self.model_names[-1] + '.hdf5')
            print('Creating keras model from file {}'.format(
                path
            ))
            self.m = keras.models.load_model(path)


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
        lr = keras.backend.eval(self.m.optimizer.lr)

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
            ax1.plot(
                self.df.epoch, _smooth(ys, 2),
                lw=1, alpha=1, ls='-', c=color, zorder=3,
                label='{} (max={})'.format(label, ys.max()),
            )
            ax1.fill_between(
                self.df.epoch,
                # np.maximum.accumulate(ys),
                # np.minimum.accumulate(ys[::-1])[::-1],
                _smooth(np.maximum.accumulate(ys), 1),
                _smooth(np.minimum.accumulate(ys[::-1])[::-1], 1),
                facecolor=color,
                alpha=1/3,
                zorder=0,
            )

        def _steps():
            i = 0
            start = 0
            step = 3
            while start <= border_dist.max():
                yield i, start, start + step
                start += step
                step += 1
                i += 1

        def _kernel(radius, strength):
            a = np.zeros(radius * 2 + 1, float)
            a[radius] = 1
            a = ndi.gaussian_filter1d(a, radius / (4 / strength))
            return a

        def _smooth(y, strength):
            res = np.zeros(y.shape, dtype=float)
            for radius, start, end in _steps():
                kernel = _kernel(radius, strength)
                res += (
                    ndi.filters.convolve1d(y, kernel, mode='nearest') *
                    ((border_dist >= start) & (border_dist < end))
                )
            return res

        if self.df.shape[-1] < 2:
            print('No plot when less than 2 elements!')
            return

        border_dist = np.min(
            np.c_[self.df.epoch, self.df.epoch.max() - self.df.epoch],
            axis=1,
        )

        fig, ax1 = plt.subplots(figsize=(16, 9))
        _show_accuracy(self.df.acctrain, 'red', 'Accuracy train set')
        _show_accuracy(self.df.acctest, 'green', 'Accuracy test set')
        ax1.plot(
            self.df.epoch, self.df.lr,
            lw=2, alpha=1/3, ls='-', c='orange', label='Learning rate', zorder=1
        )
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('percent')
        plt.legend(loc='lower center')

        ax2 = ax1.twinx()
        mask = self.df.loss < self.df.loss.min() + self.df.loss.std() * 3
        ax2.plot(
            self.df.epoch[mask],
            self.df.loss[mask],
            lw=1, alpha=1, ls='-', c='purple',
            label='loss (min={:.8f})'.format(self.df.loss.min()),
            zorder=2,
        )
        ax2.set_ylabel('loss', color='purple')
        ax2.tick_params('y', colors='purple')
        plt.legend(loc='lower left')

        plt.tight_layout()

        plt.savefig(
            self.directory + '/status.png',
            dpi=180, orientation='landscape', bbox_inches='tight',
        )
