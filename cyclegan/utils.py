import random, torch, itertools
import numpy as np

class ReplayBuffer:
    """Discriminator'i gecmis uretilmis goruntulerle stabilize eder."""
    def __init__(self, max_size=50):
        self.max_size = max_size; self.data = []
    def push_pop(self, data):
        out = []
        for el in data:
            el = el.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(el); out.append(el)
            elif random.random() > 0.5:
                i = random.randint(0, self.max_size - 1)
                out.append(self.data[i].clone()); self.data[i] = el
            else:
                out.append(el)
        return torch.cat(out)

class LambdaLR:
    """Belirli epoch'tan sonra lineer azalan ogrenme orani."""
    def __init__(self, n_epochs, decay_start):
        self.n_epochs = n_epochs; self.decay_start = decay_start
    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start) / float(self.n_epochs - self.decay_start)
