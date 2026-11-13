"""Validation-based stopping, independent from best-checkpoint selection."""
import math


class EarlyStopping:
    def __init__(self, settings, state=None):
        self.patience, self.min_delta = settings['patience'], settings['min_delta']
        self.best, self.bad_validations = -float('inf'), 0
        if state:
            self.best, self.bad_validations = state['best'], state['bad_validations']

    def update(self, score):
        if not math.isfinite(score):
            raise FloatingPointError('Nonfinite validation selection score')
        if score > self.best + self.min_delta:
            self.best, self.bad_validations = score, 0
        else:
            self.bad_validations += 1
        return bool(self.patience and self.bad_validations >= self.patience)

    def state_dict(self):
        return dict(best=self.best, bad_validations=self.bad_validations)
