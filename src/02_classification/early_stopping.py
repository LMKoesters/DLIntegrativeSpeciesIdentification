class EarlyStopping:
    def __init__(self, patience, delta=.0):
        self.patience = patience
        self.counter = 1
        self.val_loss_min = None
        self.early_stop = False
        self.delta = delta
        self.best_epoch = False

    def __call__(self, val_loss):
        # if first epoch
        if self.val_loss_min is None:
            self.val_loss_min = val_loss
            self.best_epoch = True
        # if validation loss did not get smaller
        elif val_loss >= (self.val_loss_min + self.delta):
            self.counter += 1
            if self.counter == self.patience:
                self.early_stop = True
            self.best_epoch = False
        # if validation loss got smaller :)
        else:
            self.val_loss_min = val_loss
            self.counter = 1
            self.best_epoch = True

