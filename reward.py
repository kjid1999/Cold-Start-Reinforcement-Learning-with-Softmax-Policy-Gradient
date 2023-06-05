import torch
from torch.nn.functional import softmax
from torch.distributions import Categorical

class Updater:
    def __init__(self, model, optimizer, R, p_drop=0.5, W=10000, J=1) -> None:
        self.p_drop = p_drop
        self.W = W
        self.J = J
        self.optimizer = optimizer
        self.R = R
        self.model = model

        
    def updata(self, x, y):

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