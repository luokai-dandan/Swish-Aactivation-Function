# -*- coding: UTF-8 -*-
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self, beta):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        x = x * torch.sigmoid(self.beta*x)
        return x

def Curve01(x, beta:list):

    swish0 = Swish(beta[0])
    y0 = swish0(x)
    swish1 = Swish(beta[1])
    y1 = swish1(x)
    swish2 = Swish(beta[2])
    y2 = swish2(x)
    swish3 = Swish(beta[3])
    y3 = swish3(x)
    plt.title('Swish')
    plt.plot(x, y0, color='green', label='SMU,β=0.1')
    plt.plot(x, y1, color='blue', label='SMU,β=1')
    plt.plot(x, y2, color='red', label='SMU,β=5')
    plt.plot(x, y3, '--', color='skyblue', label='SMU,β=10')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.8)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    x = torch.linspace(-3, 3, 10000)
    beta = [0.1, 1, 5, 10]
    Curve01(x, beta=beta)
