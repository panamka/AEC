from numpy import log10 as lg
import torch

def to_db(ratio):
    assert ratio >= 0
    return 10. * lg(ratio + 1e-8)

def from_db(ratio_db, base=10):
    return 10 ** (ratio_db / base) - 1e-8

def get_coef(db_cur, db_target):
    diff = db_cur - db_target
    mult = from_db(diff / 2, base=10)
    return mult

def normal(mean=0, std=1, *, generator=None):
    x = torch.empty(1).normal_(mean, std, generator=generator).item()
    return x