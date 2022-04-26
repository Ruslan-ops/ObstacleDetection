# def Callback():
#     def __init__(self): pass
#     def on_train_begin(self): pass
#     def on_train_end(self): pass
#     def on_epoch_begin(self): pass
#     def on_epoch_end(self): pass
#     def on_batch_begin(self): pass
#     def on_batch_end(self): pass
#     def on_loss_begin(self): pass
#     def on_loss_end(self): pass
#     def on_step_begin(self): pass
#     def on_step_end(self): pass

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0, max_train_diff=0.25):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.max_train_diff = max_train_diff
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss, train_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            train_diff = (val_loss - train_loss)/val_loss
            #print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience and train_diff > self.max_train_diff:
                print(f'INFO: Early stopping. train_diff={train_diff}')
                self.early_stop = True
        return self.early_stop