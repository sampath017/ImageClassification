from colorama import Fore, Style
import torch
import wandb
from torch.utils.data import DataLoader
from pathlib import Path
from callbacks import OverfitCallback, EarlyStoppingCallback
import time
from dataset import MapDataset
import torch.optim.lr_scheduler as lrs


class Trainer:
    def __init__(
        self,
        module=None,
        logger=None,
        callbacks=[],
        logs_path=None,
        optimizer=None,
        lr_scheduler=None,
        limit_train_batches=None,
        limit_val_batches=None,
        fast_dev_run=False,
        measure_time=False,
        num_workers=6,
        save_checkpoint_type="every"
    ):
        self.module = module

        self.logger = logger
        if self.logger.mode == "online":
            if fast_dev_run:
                self.logger.mode = "offline"

        self.logs_path = logs_path
        self.optimizer = optimizer
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.lr_scheduler = lr_scheduler
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.fast_dev_run = fast_dev_run
        self.measure_time = measure_time
        self.num_workers = num_workers
        self.save_checkpoint_type = save_checkpoint_type

        self.overfit_callback = None
        self.training_step = 0
        self.validation_step = 0
        self._earlystopping_callback()

        self.step_lr_schedulers = [lrs.OneCycleLR]
        self.epoch_lr_schedulers = []

        print(f"Using device: {module.device}!")

    def setup(self):
        self.logger.init()
        wandb.log({"model_architecture": self.module.model})
        self.max_epochs = self.logger.config["max_epochs"]

        # Fast dev run
        if self.fast_dev_run:
            self.limit_train_batches = 5
            self.limit_val_batches = 5
            self.max_epochs = 1
        # Overfit callback
        else:
            self.overfit_callback = self._overfit_callback()
            if self.overfit_callback:
                self.save_checkpoint_type = None
                if self.overfit_callback.augument_data:
                    train_dataset = self.train_dataloader.dataset
                else:
                    train_dataset = MapDataset(
                        self.train_dataloader.dataset, transform=self.val_dataloader.dataset.transform)

                self.train_dataloader = DataLoader(
                    train_dataset, batch_size=self.overfit_callback.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
                self.limit_train_batches = self.overfit_callback.limit_train_batches
                self.limit_val_batches = self.overfit_callback.limit_val_batches
                self.max_epochs = self.overfit_callback.max_epochs

        # lr_schedular
        # step lrs
        for scheduler_class in self.step_lr_schedulers:
            if isinstance(self.lr_scheduler, scheduler_class):
                self.lr_scheduler_on_epoch = False
                break

        # epoch lrs
        for scheduler_class in self.epoch_lr_schedulers:
            if isinstance(self.lr_scheduler, scheduler_class):
                self.lr_scheduler_on_epoch = True
                break

    def _overfit_callback(self):
        for callback in self.callbacks:
            if isinstance(callback, OverfitCallback):
                return callback

    def _earlystopping_callback(self):
        for callback in self.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                self.early_stopping_callback = callback
                break
            else:
                self.early_stopping_callback = False

    def _earlystopping_callback_check(self, epoch_train_accuracy, epoch_val_accuracy):
        stop_training = False
        accuracy_diff = None
        if self.early_stopping_callback:
            stop_training, accuracy_diff = self.early_stopping_callback.check(
                epoch_train_accuracy, epoch_val_accuracy)

        return stop_training, accuracy_diff

    def _lr_scheduler_update(self, on_epoch=True):
        if on_epoch:
            wandb.log({
                "epoch": self.epoch,
                "lr": self.current_lr
            })
        else:
            wandb.log({
                "training_step": self.training_step,
                "lr": self.current_lr
            })

        if self.lr_scheduler:
            self.lr_scheduler.step()

    def fit(self, train_dataloader, val_dataloader):
        # setup
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.setup()
        best_epoch_val_accuracy = 0.0

        for epoch in range(self.max_epochs):
            self.current_lr = self.optimizer.param_groups[0]['lr']
            measure_time_bool = self.measure_time and epoch == 0
            self.epoch = epoch

            # Train
            if measure_time_bool:
                start_time = time.time()
            epoch_train_loss, epoch_train_accuracy = self.train()
            if measure_time_bool:
                end_time = time.time()

            # lr_scheduler step
            if self.lr_scheduler and self.lr_scheduler_on_epoch:
                self._lr_scheduler_update(on_epoch=True)

            if measure_time_bool:
                # type: ignore
                print(f"Time per epoch: {end_time-start_time:.2f} seconds")

            if not self.overfit_callback:
                epoch_val_loss, epoch_val_accuracy = self.val(
                    val_dataloader)
            else:
                epoch_val_accuracy = torch.inf

            # Early Stopping
            stop_training, accuracy_diff = self._earlystopping_callback_check(
                epoch_train_accuracy, epoch_val_accuracy)

            # Print
            # fmt:off
            if self.early_stopping_callback:
                color = Fore.RED if self.early_stopping_callback.counted else ""
                reset = Style.RESET_ALL if color else ""  
            else:
                color = ""
                reset = ""

            print(f"Epoch: {self.epoch}, train_accuracy: {\
                epoch_train_accuracy:.2f}, val_accuracy: {color}{epoch_val_accuracy:.2f}{reset}, lr: {self.current_lr:.4f}")

            if self.fast_dev_run:
                print("Sanity check done with fast dev run!")

            if hasattr(self, 'overfit_callback') and self.overfit_callback and epoch_train_accuracy >= 100.0:
                print(f"Overfit done at epoch: {epoch}.")
                break
            # fmt:on

            if self.save_checkpoint_type == "every":
                self.save_checkpoint()
            elif self.save_checkpoint_type == "best_val":
                if epoch_val_accuracy > best_epoch_val_accuracy:
                    best_epoch_val_accuracy = epoch_val_accuracy
                    self.save_checkpoint(
                        best_epoch_val_accuracy, save_best_model=True)

            if stop_training:
                # fmt:off
                print(f"Stoppping training due to early stopping crossing threshold {\
                    accuracy_diff:.2f}")
                # fmt:on
                break

        if epoch == self.max_epochs:
            # fmt:off
            print(f"Training stopped max_epochs: {self.max_epochs} reached!")
            # fmt:on

        if not self.fast_dev_run and self.save_best_model:
            wandb.log_model(self.save_path, aliases=["best"])

    def train(self):
        step_train_losses = []
        step_train_accuracies = []

        self.module.model.train()
        for step, batch in enumerate(self.train_dataloader):
            if self.limit_train_batches and (step > self.limit_train_batches):
                break

            self.training_step += 1

            loss, acc = self.module.training_step(batch)
            step_train_loss = loss.item()
            step_train_accuracy = acc.item()
            step_train_losses.append(step_train_loss)
            step_train_accuracies.append(step_train_accuracy)
            wandb.log({
                "training_step": self.training_step,
                "step_train_loss": step_train_loss,
                "step_train_accuracy": step_train_accuracy
            })

            # optimization
            self.optimizer.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # lr_scheduler step
            if self.lr_scheduler and not self.lr_scheduler_on_epoch:
                self._lr_scheduler_update(on_epoch=False)

        epoch_train_loss = torch.tensor(step_train_losses).mean()
        epoch_train_accuracy = torch.tensor(step_train_accuracies).mean()

        wandb.log({
            "epoch": self.epoch,
            "epoch_train_loss": epoch_train_loss,
            "epoch_train_accuracy": epoch_train_accuracy,
        })

        return epoch_train_loss, epoch_train_accuracy

    @torch.no_grad()
    def val(self, val_dataloader):
        step_val_losses = []
        step_val_accuracies = []

        self.module.model.eval()
        for step, batch in enumerate(val_dataloader):
            self.validation_step += 1
            if self.limit_val_batches and (step > self.limit_val_batches):
                break

            loss, acc = self.module.validation_step(batch)
            step_val_loss = loss.item()
            step_val_accuracy = acc.item()
            step_val_losses.append(step_val_loss)
            step_val_accuracies.append(step_val_accuracy)
            wandb.log({
                "validation_step": self.validation_step,
                "step_val_loss": step_val_loss,
                "step_val_accuracy": step_val_accuracy
            })

        epoch_val_loss = torch.tensor(step_val_losses).mean()
        epoch_val_accuracy = torch.tensor(step_val_accuracies).mean()

        wandb.log({
            "epoch": self.epoch,
            "epoch_val_loss": epoch_val_loss,
            "epoch_val_accuracy": epoch_val_accuracy,
        })

        return epoch_val_loss, epoch_val_accuracy

    def save_checkpoint(self, val_accuracy=None, save_best_model=False):
        self.save_best_model = save_best_model
        self.checkpoint_path = Path(wandb.run.dir).parent / "checkpoints"
        self.checkpoint_path.mkdir(exist_ok=True)
        if save_best_model:
            self.save_path = self.checkpoint_path / \
                f"best_val_acc_{val_accuracy:.2f}.pt"
        else:
            self.save_path = self.checkpoint_path / \
                f"checkpoint-{self.epoch}.pt"

        state_dict = {
            "model": self.module.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        if self.lr_scheduler:
            state_dict["lr_scheduler"] = self.lr_scheduler.state_dict()

        # Remove previous saved models
        for file in self.save_path.parent.iterdir():
            file.unlink()

        torch.save(state_dict, self.save_path)

        if not save_best_model:
            wandb.log_model(self.save_path, aliases=[
                            f"[checkpoint-{self.epoch}]"])
