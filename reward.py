import torch
from torch.nn.functional import softmax
from torch.distributions import Categorical

def DUP(z, voc):
    out = torch.zeros_like(voc)
    out[voc == z[-1]] = -1
    return out

def EOS(voc, t, target_sentence, end_token):
    out = torch.zeros_like(voc)
    if t < len(target_sentence):
        # voc == end_token 也許再改
        out[voc == end_token] = -1
    return out

class Updater:
    def __init__(self, model, optimizer, R: function, p_drop=0.5, W=10000, J=1) -> None:
        self.p_drop = p_drop
        self.W = W
        self.J = J
        self.optimizer = optimizer
        self.R = R
        self.model = model

    def updata(self, x, y):
        L_BBSPG = 0
        for j in range(1, J+1):
            z = []
            for t in range(1, self.T+1):
                mu = torch.rand((1, ))

                model_output = self.model()
                if mu > p_drop:
                    # 參數傳入也許再改
                    delta_r = W * (self.R(z, voc, t, y) - self.R(z, voc, t-1, y) + DUP(z, voc) + EOS(voc, t, y, eos))

                    prob = softmax(torch.log(model_output) + delta_r)
                    zt_idx = Categorical(probs=prob).sample()
                    z_t = voc[zt_idx]

                    L_BBSPG = L_BBSPG.detach()
                    L_BBSPG -= torch.log(model_output[zt_idx])
                    L_BBSPG.backward()
                
                else:
                    prob = model_output
                    z_t = voc[Categorical(probs=prob).sample()]

                z.append(z_t)
        return

p_drop=0.5
W=10000
J=1

def f():
    for j in range(1, J+1):
        z = []
        for t in range(1, T+1):
            mu = torch.rand((1, ))

            model_output = model()
            if mu > p_drop:
                delta_r = W * (R() - R() + DUP() + EOS())

                prob = softmax(torch.log(model_output) + delta_r)
                z_t = Categorical(probs=prob).sample()

                L_BBSPG -= model_output
                L_BBSPG.backward()
            
            else:
                prob = model_output
                z_t = Categorical(probs=prob).sample()

            z.append(z_t)