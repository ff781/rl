import torch
import numpy as np
import random


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class M:

    def __init__(
        self,
        count=0,
        sum=None,
        value=None,
        var_unnormalized=0.,
        min=None,
        max=None,
        track_extremes=False,
        track_variance=False,
        alpha_ema=None,
        alpha_emv=None,
        tensor_reduction=None,
    ):
        self.track_extremes = track_extremes
        self.track_variance = track_variance
        self.alpha_ema = alpha_ema
        self.alpha_emv = alpha_emv
        self.tensor_reduction = tensor_reduction

        if value is not None:
            sum = count * value
        self.count = count
        self.sum = sum
        self.min = min
        self.max = max
        self._ema = None
        self._emv = None
        self._var_unnormalized = var_unnormalized
        self.recent_value = None

    def add(self, value, count=1):
        if hasattr(value, 'tolist') and hasattr(value, 'mean'):
            if self.tensor_reduction is None:
                value = value
            elif self.tensor_reduction == 'mean':
                value = value.mean()
        if isinstance(value, list):
            for v in value:
                self.add(v)
            return

        if count == 0:
            return

        if self.track_variance:
            old_mean = self.mean()

        if value is None:
            value = self.mean(default=0)

        if self.sum is None:
            self.sum = value * count
        else:
            self.sum = self.sum + value * count
        self.count += count
        self.recent_value = value

        if self.track_extremes:
            self.update_extremes(value)

        if self.track_variance:
            if old_mean is not None:
                mean = self.mean()
                self._var_unnormalized += count * (value - old_mean) * (value -
                                                                        mean)

        if self.alpha_ema is not None:
            if self._ema is None:
                self._ema = value
            else:
                self._ema = (
                    1 - self.alpha_ema) * value + self.alpha_ema * self._ema

        if self.alpha_emv is not None:
            mean = self.mean()
            if self._emv is None:
                self._emv = (value - self._ema)**2
            else:
                self._emv = (1 - self.alpha_emv) * (
                    value - mean)**2 + self.alpha_emv * self._emv

    def add_batch(self, values):
        if isinstance(values, (torch.Tensor, np.ndarray)):
            batch_size = values.shape[0]
            values_flat = values.reshape((batch_size, -1) if values.ndim > 1 else (batch_size,))
            values_sum = values_flat.sum(0)
            values_count = values_flat.shape[0]
        else:
            raise TypeError(
                "Unsupported type for batch addition. Use torch.Tensor or np.ndarray."
            )

        if self.sum is None:
            self.sum = values_sum
        else:
            self.sum = self.sum + values_sum

        self.count += values_count

        if self.track_extremes:
            values_min = meth.min(values_flat, 0)
            values_max = meth.max(values_flat, 0)
            self.update_extremes(values_min=values_min, values_max=values_max)

        if self.track_variance:
            old_mean = self.mean()
            new_mean = self.sum / self.count
            self._var_unnormalized += ((values_flat - old_mean)**2).sum()

        if self.alpha_ema is not None:
            if self._ema is None:
                self._ema = values_sum / values_count
            else:
                self._ema = (1 - self.alpha_ema) * (
                    values_sum / values_count) + self.alpha_ema * self._ema

        if self.alpha_emv is not None:
            mean = self.mean()
            if self._emv is None:
                self._emv = ((values_flat - mean)**2).mean()
            else:
                self._emv = (1 - self.alpha_emv) * ((
                    values_flat - mean)**2).mean() + self.alpha_emv * self._emv

    def update_extremes(self, values=None, values_min=None, values_max=None):
        if values is not None:
            values_min = values_max = values
        if values_min is not None:
            if self.min is None:
                self.min = values_min
            else:
                _ = meth.lowest_lib(self.min, values_min)
                if _ is torch:
                    self.min = torch.minimum(self.min, values_min)
                else:
                    self.min = np.minimum(self.min, values_min)
        if values_max is not None:
            if self.max is None:
                self.max = values_max
            else:
                _ = meth.lowest_lib(self.max, values_max)
                if _ is torch:
                    self.max = torch.maximum(self.max, values_max)
                else:
                    self.max = np.maximum(self.max, values_max)

    def __add__(self, other):
        r = copy.copy(self)
        r @ other
        return r

    def __mul__(self, other):
        self.sum *= other

    def __truediv__(self, other):
        self.sum /= other

    def __matmul__(self, other):
        if isinstance(other, tuple):
            self.add(other[0], count=other[1])
        elif isinstance(other, list):
            for value in other:
                self @ value
        elif isinstance(other, M):
            self.add(other.mean(), other.count)
            if self.track_extremes:
                self.update_extremes(values_min=other.min,
                                     values_max=other.max)
        else:
            self.add(other, 1)
        return self

    def mean(self, default=None):
        return default if self.sum is None else self.sum / self.count if self.count != 0 else default

    def var(self, ddof=1, default=None):
        if not self.track_variance:
            raise AttributeError("Variance tracking is not enabled.")
        try:
            return self._var_unnormalized / (self.count - ddof)
        except ZeroDivisionError:
            return default

    def std(self, ddof=1, default=None):
        v = self.var(ddof=ddof, default=None)
        if v is None:
            return default
        return v.sqrt() if hasattr(v, 'sqrt') else np.sqrt(v)

    def ema(self, default=None):
        return self._ema if self._ema is not None else default

    def emv(self, default=None):
        return self._emv if self._emv is not None else default

    def __float__(self):
        return float(self.mean('nan'))

    def __repr__(self):
        r = f'μ{self.mean()} '
        if self.track_variance:
            r += f'σ{self.std()} '
        if self.track_extremes:
            r += f'[{self.min}, {self.max}] '
        if self.alpha_ema is not None:
            r += f'dμ{self.ema()} '
        if self.alpha_emv is not None:
            r += f'dσ{self.emv()} '
        r += f'n{self.count}'
        return r

    def to_dict(self):
        r = dict(
            mean=self.mean(),
            count=self.count,
        )
        if self.track_variance:
            r['std'] = self.std()
        if self.track_extremes:
            r['min'] = self.min
            r['max'] = self.max
        if self.alpha_ema is not None:
            r['ema'] = self.ema()
        if self.alpha_emv is not None:
            r['emv'] = self.emv()
        return r
