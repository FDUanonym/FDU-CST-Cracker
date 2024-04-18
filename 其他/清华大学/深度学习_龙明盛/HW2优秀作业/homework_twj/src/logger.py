import shutil
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    def __init__(self, log_dir):
        shutil.rmtree(log_dir)
        self.writer = SummaryWriter(log_dir)
        self.global_step = 1

    def log_value(self, name, value):
        self.writer.add_scalar(name, value, global_step=self.global_step)
        return self

    def step(self):
        self.global_step += 1