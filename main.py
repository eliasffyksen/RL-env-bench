import torch as t
import torch.nn as nn
from torch.utils.cpp_extension import load
from typing_extensions import Final
from timeit import timeit
from tqdm import tqdm
import matplotlib.pyplot as plt

print('building crun extension')
crun = load(
  name='crun',
  sources=['./crun.cpp'],
)

print('build models')
class Actor(t.nn.Module):
  dist: t.Tensor

  def __init__(self, dist):
    super().__init__()
    self.dist = dist

  def forward(self, x):
    return self.dist

class Environment(t.nn.Module):
  board_size: int
  ACTIONS: t.Tensor

  def __init__(self, board_size, actor):
    super().__init__()
    self.actor = actor
    self.softmax = nn.Softmax(0)
    self.board_size = board_size
    self.ACTIONS = t.tensor([
      0, 0, 1, 0, -1, 0
    ])

  def forward(self, batch_size):
    # type: (int) -> t.Tensor
    states = t.zeros((
      batch_size,
      self.board_size,
      self.board_size
    ))
    state = t.randint(0, self.board_size - 1, (2,), dtype=t.long)

    for i in range(batch_size):
      states[
        i, state[0], state[1]
      ] = 1

      action_dist = self.actor(states[i])
      action_dist = self.softmax(action_dist)
      action = action_dist.multinomial(1)[0]

      state += self.ACTIONS[action:action + 2]
      state = state.clamp(0, self.board_size - 1)

    return states

actor = Actor(t.tensor([1,1,1,1,1], dtype=t.float))
env = Environment(8, actor)
env_jit = t.jit.script(env)
t.jit.save(env_jit, 'models/env.pt')
if not crun.load('models/env.pt'):
  raise Exception('Failed to load model through crun extension')

env_labels = [
  'Raw python',
  'local jit from python',
  'local jit from C++ extension',
  'file jit from C++ extension',
  'file jit from C++ executable',
]

envs = [
  env,
  env_jit,
  lambda batch_size: crun.run_model(env_jit._c, batch_size),
  lambda batch_size: crun.run(batch_size),
]
env_times = [[] for _ in envs]
cpp_times_us = [
        131,
        247,
        520,
        652,
        1348,
        2733,
        5001,
        9928,
        20910,
        44725,
        88648,
        175050,
]
env_times.append([ x / (10 ** 6) for x in cpp_times_us])

print('initializing')
for _ in range(10):
  for env in envs:
    env(10)

print('testing')
batch_sizes = []
batch_size = 2
for i in tqdm(range(12)):
  batch_size *= 2
  batch_sizes.append(batch_size)
  for j, env in enumerate(envs):
    time = timeit(lambda: env(batch_size), number=1)
    env_times[j].append(time)

print(env_times)
print(batch_sizes)
plt.figure(figsize=(20,10))
for i, env_label in enumerate(env_labels):
  plt.plot(batch_sizes, env_times[i], label=env_label)
plt.xlabel('Batch size')
plt.ylabel('Exec time (sec)')
plt.legend()
plt.show()