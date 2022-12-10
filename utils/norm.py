import torch
from dataclasses import dataclass

import utils.db_utils as db_utils


@dataclass
class Amplifier:
    mult: float

    def __call__(self, x):
        return x * self.mult

class Evaluator:
    def __call__(self, x):
        mean_sq = (x **2).mean()
        db = db_utils.to_db(mean_sq)
        return db

class Factory:
    def __init__(self, evaluator, default_target_level):
        self.evaluator = evaluator
        self.default_target_level = default_target_level

    def __call__(self, ref_sig, target_level):
        cur_level = self.evaluator(ref_sig)
        mult = db_utils.get_coef(cur_level, target_level)
        transform = Amplifier(mult)
        return transform

class Transform:
    def __init__(self,
                 speech_level_db=-28,
                 ns_level_db=-38
                 ):
        self.speech_level_db = speech_level_db
        self.ns_level_db = ns_level_db
        self.norm_factory = Factory(
            evaluator=Evaluator(),
            default_target_level=-25,
        )

    def __call__(self, utt):
        normalizer_utt = self.norm_factory(utt, target_level=db_utils.normal(self.speech_level_db, 1))
        utt = normalizer_utt(utt)

        #normalizer_ns = self.norm_factory(ns, target_level=db_utils.normal(self.speech_level_db, 1))
        #ns = normalizer_ns(ns)
        #mixture = utt + ns
        return utt