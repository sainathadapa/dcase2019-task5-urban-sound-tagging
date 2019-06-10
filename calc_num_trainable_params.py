from utils import Task5Model
import numpy as np

def calc_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = Task5Model(31)
print(calc_num_params(model.bw2col) + calc_num_params(model.mv2.features) + calc_num_params(model.final))

model = Task5Model(37)
print(calc_num_params(model.bw2col) + calc_num_params(model.mv2.features) + calc_num_params(model.final))

