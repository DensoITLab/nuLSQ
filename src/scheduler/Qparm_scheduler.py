import math



class CosineAnnealingScheduler(object):

    def __init__(self, model, key, init_coeff, T_max, eta_min=0, last_epoch=-1):
        self.init_coeff = init_coeff
        self.model = model
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.key = key
        super(CosineAnnealingScheduler, self).__init__()

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
            self.curr_coeff = self.get_coeff ()

        print("coeff in Cos damp=", self.curr_coeff)

    def get_coeff(self):
        if self.last_epoch == 0:
            return self.init_coeff
        elif self.last_epoch > 0:
            return self.eta_min + (self.init_coeff - self.eta_min) * \
                    (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2


class LinearWarmupCosineAnnealingScheduler(object):

    def __init__(self, model, key, init_coeff, T_max, const_coeff=0, eta_min=0, const_epoch = 10, last_epoch=-1):
        self.init_coeff = init_coeff
        self.model = model
        self.key = key
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.const_epoch = const_epoch
        self.const_coeff = const_coeff
        super(LinearWarmupCosineAnnealingScheduler, self).__init__()

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
            if self.last_epoch < self.const_epoch:
                self.curr_coeff = self.const_coeff \
                + (self.last_epoch + 1)* (self.init_coeff - self.const_coeff)/ (self.const_epoch )

            else:
                self.curr_coeff = self.get_coeff ()

        # print("coeff in LinearWarmup=", self.loss.weighting)

    def get_coeff(self):
        if (self.last_epoch - self.const_epoch) == 0:
            return self.init_coeff
        elif (self.last_epoch - self.const_epoch) > 0:
            return self.eta_min + (self.init_coeff - self.eta_min) * \
                    (1 + math.cos((self.last_epoch - self.const_epoch) * math.pi / (self.T_max - self.const_epoch))) / 2
        elif ((self.last_epoch - self.const_epoch) - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self.curr_coeff + (self.init_coeff - self.eta_min) * \
                    (1 - math.cos(math.pi / (self.T_max - self.const_epoch))) / 2
        # recurrsive definition if needed
        # return (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / \
        #         (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) * \
        #         (self.curr_coeff - self.eta_min) + self.eta_min


def scheduler_class(scheduler_name, model, key,  coeff, epochs):
    if scheduler_name == "CosineAnnealing":
        scheduler = CosineAnnealingScheduler(model, key, init_coeff = coeff, T_max= epochs )
    elif scheduler_name == "LinearWarmupCosineAnnealing":
        scheduler = LinearWarmupCosineAnnealingScheduler(model, key,  init_coeff = coeff, T_max= epochs )
    else:
        print(scheduler_name,": Non-Implemented Scheduler")
        scheduler = None
    return scheduler

