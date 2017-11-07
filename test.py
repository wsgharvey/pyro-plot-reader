import torch
from torch.autograd import Variable

import pyro
import pyro.infer
import pyro.distributions as dist


def weather(temp):
    cloudy = pyro.sample('cloudy', dist.bernoulli,
                         Variable(torch.Tensor([0.3])))
    cloudy = 'cloudy' if cloudy.data[0] == 1.0 else 'sunny'
    mean_temp = {'cloudy': [55.0], 'sunny': [75.0]}[cloudy]
    sigma_temp = {'cloudy': [10.0], 'sunny': [15.0]}[cloudy]
    temp = pyro.sample('temp', dist.normal,
                       Variable(torch.Tensor(mean_temp)),
                       Variable(torch.Tensor(sigma_temp)))
    return cloudy, temp.data[0]


posterior = pyro.infer.Importance(weather, num_samples=100)
print(posterior)
marginal = pyro.infer.Marginal(posterior)

print("unconditioned:\n", marginal(Variable(torch.Tensor([75]))))


conditioned_weather = pyro.condition(
                        weather,
                        data={'temp': Variable(torch.Tensor([75]))}
)

posterior = pyro.infer.Importance(conditioned_weather, num_samples=100)
marginal = pyro.infer.Marginal(posterior)

for _ in range(100):
    print("conditioned:\n", marginal(Variable(torch.Tensor([75]))))
