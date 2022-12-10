from torch_stoi import NegSTOILoss as stoi
from torchmetrics.audio import SignalDistortionRatio


def calc_stoi(pred, clean, sr=16000):
    loss_func = stoi(sample_rate=sr)
    return loss_func(pred.detach().cpu(), clean.detach().cpu())

def calc_sdr(pred, clean):
    sdr = SignalDistortionRatio()
    return sdr(pred.detach().cpu(), clean.detach().cpu())


# def calc_sdr(pred, clean, tol=1e-4):
#     pred_l2 = (pred ** 2).mean(-1, keepdim=True) ** 0.5
#     clean_l2 = (clean ** 2).mean(-1, keepdim=True) ** 0.5
#
#     pred_l2[pred_l2 < tol] = tol
#     clean_l2[clean_l2 < tol] = tol
#
#     pred_norm = pred / pred_l2
#     clean_norm = clean / clean_l2






