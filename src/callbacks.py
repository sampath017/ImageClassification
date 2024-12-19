class OverfitCallback:
    def __init__(self, limit_batches=2, batch_size=None, limit_train_batches=2, limit_val_batches=2, max_epochs=200, augument_data=True):
        if limit_batches > 0:
            self.limit_train_batches = limit_batches
            self.limit_val_batches = limit_batches
        else:
            self.limit_train_batches = limit_train_batches
            self.limit_val_batches = limit_val_batches

        self.max_epochs = max_epochs
        self.augument_data = augument_data
        self.batch_size = batch_size


class EarlyStoppingCallback:
    def __init__(self, wait_epochs=3, accuracy_diff=2.0, min_val_accuracy=90.0):
        if wait_epochs < 0:
            raise ValueError("wait_epochs should be > 0!")

        self.wait_epochs = wait_epochs
        self.accuracy_diff = accuracy_diff
        self.waited_epochs = 0
        self.min_val_accuracy = min_val_accuracy
        self.check_early_stopping = False
        self.counted = False

    def check(self, epoch_train_accuracy, epoch_val_accuracy):
        self.counted = False
        stop_training = False
        if not self.check_early_stopping and (epoch_val_accuracy >= self.min_val_accuracy):
            print("Early stopping check started!")
            self.check_early_stopping = True

        if self.check_early_stopping:
            diff = epoch_train_accuracy - epoch_val_accuracy
            if diff > self.accuracy_diff:
                self.waited_epochs += 1
                self.counted = True
                if self.waited_epochs >= self.wait_epochs:
                    stop_training = True

        return stop_training, self.accuracy_diff
