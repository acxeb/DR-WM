import datetime
import time
import os
from collections import defaultdict, deque
from tqdm import tqdm
import torch
import sys
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def redirect_stdout_to_file(epoch = None, log_dir="logs", is_train=True):
    os.makedirs(log_dir, exist_ok=True)
    if is_train:
        log_path = os.path.join(log_dir, f"epoch_{epoch:03d}.txt")
    else:
        log_path = os.path.join(log_dir, f"{epoch}.txt")
    return open(log_path, "w")

class StringValue(object):
    def __init__(self, initial=""):
        self.value = str(initial)

    def update(self, value, n=1):
        self.value = str(value)

    def __str__(self):
        return self.value

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        if not (value != value):
            self.count += n
            self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t", max_str_len=16):
        self.meters = {}
        self.delimiter = delimiter
        self.max_str_len = max_str_len

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, (float, int)):
                meter = self.meters.get(k)
                if meter is None or not isinstance(meter, SmoothedValue):
                    meter = SmoothedValue()
                    self.meters[k] = meter
                meter.update(v)
            else:
                meter = self.meters.get(k)
                if meter is None or not isinstance(meter, StringValue):
                    meter = StringValue()
                    self.meters[k] = meter
                meter.update(v)
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if isinstance(meter, SmoothedValue):
                value_str = str(meter)
            else:
                text = str(meter)
                if len(text) < self.max_str_len:
                    for i in range(1, self.max_str_len-len(text)):
                        text = text + " "
                value_str = text
            loss_str.append("{}: {}".format(name, str(value_str)))
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0

        pbar = tqdm(iterable, total=len(iterable), desc=header, ncols=130)

        for obj in pbar:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                postfix = {
                    'eta': eta_string,
                    'time': str(iter_time),
                }
                pbar.set_postfix(postfix)

                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
                sys.stdout.flush()
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(header, total_time_str, total_time / (len(iterable)+1)))
