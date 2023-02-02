import torch
# from numpy.fft import rfft as fft
# from numpy.fft import irfft as ifft

""" Frequency Domain Adaptive Filter """
class FDAF():
  def __init__(self, M=256, hop_len=160, beta=0.9):
    super(FDAF, self).__init__()
    self.M = M
    self.hop_len = hop_len
    self.beta = beta


  def process_hop(self, x, d, mu):
    batch_size = mu.shape[0]

    device = x.get_device()

    H = torch.zeros(batch_size, self.M + 1).to(device)
    norm = torch.full((batch_size, self.M + 1), 1e-8).to(device)
    window = torch.hann_window(self.M).to(device)

    x_old = torch.zeros(batch_size, self.M).to(device)
    num_block = mu.shape[1]

    e = torch.zeros(batch_size, (num_block-1) * self.hop_len).to(device)

    for n in range(num_block-2):
      x_n = torch.cat([x_old, x[:, n * self.hop_len:n * self.hop_len + self.M]], dim=1)
      d_n = d[:, n * self.hop_len:n * self.hop_len + self.M]
      x_old = x[:, n * self.hop_len:n * self.hop_len + self.M]

      X_n = torch.fft.rfft(x_n)
      y_n = torch.fft.irfft(H * X_n)[:, self.M:]

      e_n = d_n - y_n
      e[:, n * self.hop_len:n * self.hop_len + self.M] = e_n

      e_fft = torch.cat([torch.zeros(batch_size, self.M).to(device), e_n * window], dim=1)
      E_n = torch.fft.rfft(e_fft)

      norm = self.beta * norm + (1 - self.beta) * torch.abs(X_n) ** 2

      mu_n = mu[:, n, :]

      G = mu_n * E_n / (norm + 1e-3)
      H = H + X_n.conj() * G

      h = torch.fft.irfft(H)
      h[:, self.M:] = 0
      H = torch.fft.rfft(h)
    return e


""" Frequency Domain Kalman Filter """
class FDKF():
  def __init__(self, M=256, hop_len=160, beta=0.95, sgm2u=1e-2, sgm2v=1e-6):
    super(FDKF, self).__init__()
    self.M = M
    self.hop_len = hop_len
    self.beta = beta
    self.sgm2u = sgm2u
    self.sgm2v = sgm2v

  def process_hop(self, x, d):

    batch_size = x.shape[0]
    device = x.get_device()

    Q = self.sgm2u
    R = torch.full((batch_size, self.M + 1), self.sgm2v).to(device)
    H = torch.zeros(batch_size, self.M + 1).to(device)
    P = torch.full((batch_size, self.M + 1), self.sgm2u).to(device)

    window = torch.hann_window(self.M).to(device)
    x_old = torch.zeros(batch_size, self.M).to(device)

    num_block = x.shape[1] // self.M
    e = torch.zeros(batch_size, num_block * self.M).to(device)

    for n in range(num_block):

      x_n = torch.cat([x_old, x[:, n * self.M:(n + 1) * self.M]], dim=1)
      d_n = d[:, n * self.M:(n + 1) * self.M]
      x_old = x[:, n * self.M:(n + 1) * self.M]


      X_n = torch.fft.rfft(x_n)
      y_n = torch.fft.irfft(H * X_n)[:, self.M:]
      e_n = d_n - y_n

      e_fft = torch.cat([torch.zeros(batch_size, self.M).to(device), e_n * window], dim=1)
      E_n = torch.fft.rfft(e_fft)

      R = self.beta * R + (1.0 - self.beta) * (torch.abs(E_n) ** 2)
      P_n = P + Q * (torch.abs(H))
      K = P_n * X_n.conj() / (X_n * P_n * X_n.conj() + R)
      P = (1.0 - K * X_n) * P_n

      H = H + K * E_n
      h = torch.fft.irfft(H)
      h[:, self.M:] = 0
      H = torch.fft.rfft(h)

      e[:, n * self.M:(n + 1) * self.M] = e_n

    if e.shape[1] < x.shape[1]:
      array_pad = torch.zeros(e.shape[0], x.shape[1] - e.shape[1]).to(device)
      e = torch.cat([e, array_pad], dim=1)

    return e
