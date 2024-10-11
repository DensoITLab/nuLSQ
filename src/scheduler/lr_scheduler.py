import torch.optim as optim

class LinearWarmupScheduler(object):

    def __init__(self, optimizer, min_lr=1e-4, total_epoch=5, after_scheduler=None):
        self.optimizer = optimizer
        self.total_epoch = total_epoch
        self.min_lr = min_lr
        self.after_scheduler = after_scheduler
        self._current_epoch = 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
        super(LinearWarmupScheduler, self).__init__()

    def step(self):
        if self._current_epoch < self.total_epoch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.min_lr \
                    + (self._current_epoch + 1)* (param_group['after_lr'] - self.min_lr)/ (self.total_epoch ) 
        else:
            self.after_scheduler.step()
        self._current_epoch += 1


    def state_dict(self):
        self.after_scheduler.state_dict() \
                if self._current_epoch >= self.total_epoch else None

    def load_state_dict(self, state_dict, loaded_epoch):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self._current_epoch = loaded_epoch
        if self._current_epoch  > self.total_epoch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.after_lr
            for iepoch in range(self.total_epoch, self._current_epoch):
                self.after_scheduler.step()
 
