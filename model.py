


class Model(object):
    """
    model name: time_epoch_lr_loss_accuracy_testaccuracy_lastname
    """

    def __init__(self, directory):
        self.epoch_count = ...
        self.accuracies = ...
        self.losses = ...

        self.names_train = ...
        self.names_test = ...



        pass

    def from_scratch(self, directory):
        pass

    def from_directory(self, directory):
        pass



    @property
    def xtrain(self):
        pass

    @property
    def ytrain(self):
        pass

    @property
    def xtest(self):
        pass

    @property
    def ytest(self):
        pass


    def eval_heatmap(self, y, ypred)

    def eval_model(self):

        ypredtest = self.km.predict(self.xtest, 1, 1)
