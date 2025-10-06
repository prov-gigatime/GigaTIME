import argparse
import math

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.sum_of_squares = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        self.sum_of_squares += val * val * n

        # Calculate variance and standard deviation
        if self.count > 1:
            variance = (self.sum_of_squares - (self.sum ** 2) / self.count) / (self.count - 1)
            if variance < 0:
                print(f"Warning: Negative variance detected. Variance: {variance}, Sum of squares: {self.sum_of_squares}, Sum: {self.sum}, Count: {self.count}")
                variance = 0
            self.std = math.sqrt(variance)
        else:
            self.std = 0
