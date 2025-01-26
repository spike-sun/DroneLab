import numpy as np
import torch
import gymnasium as gym
from gymnasium import Wrapper
import torch

import skrl.envs.wrappers.torch.base


class IsaacLabWrapper(skrl.envs.wrappers.torch.base.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self._reset_once = True

    def step(self, actions: torch.Tensor):
        obs, rew, terminated, trucated, info = self._env.step(actions)
        return obs, rew.unsqueeze(1), terminated.unsqueeze(1), trucated.unsqueeze(1), info

    def reset(self, seed = None, options = None):
        if self._reset_once:
            self._reset_once = False
        return self._env.reset(seed=seed, options=options)

    def render(self):
        return

    def close(self):
        self._env.close()


class FlattenObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    '''torch.Tensor版本的FlattenObservation, 第一个维度为num_envs保留'''
    def __init__(self, env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        self.flatten_observation_space = gym.spaces.flatten_space(env.observation_space)
    
    def observation(self, observation):
        return self.flatten_observation(observation, self.observation_space)
    
    @staticmethod
    def flatten_observation(obs, space: gym.spaces.Space):
            if isinstance(space, gym.spaces.Dict):
                return torch.cat([FlattenObservation.flatten_observation(obs[key], subspace) for key, subspace in space.spaces.items()], dim=-1)
            elif isinstance(space, gym.spaces.Tuple):
                return torch.cat([FlattenObservation.flatten_observation(obs, subspace) for obs, subspace in zip(obs, space.spaces)], dim=-1)
            elif isinstance(space, gym.spaces.Box):
                return obs.view(-1, np.prod(space.shape))
            elif isinstance(space, gym.spaces.Discrete):
                return obs
            else:
                raise NotImplementedError(f"Space type {type(space)} not supported")
    

def unflatten_observation(flat_obs: torch.Tensor, space: gym.spaces.Space):
    '''
    obs: (-1, size(shape))
    space: single observation space (*shape)
    '''
    
    batch_size = flat_obs.shape[0]
    num_observations = flat_obs.shape[1]

    index = 0
    def _unflatten(space):
        nonlocal index
        if isinstance(space, gym.spaces.Dict):
            return {key: _unflatten(subspace) for key, subspace in space.spaces.items()}
        elif isinstance(space, gym.spaces.Tuple):
            return tuple(_unflatten(subspace) for subspace in space.spaces)
        elif isinstance(space, gym.spaces.Box):
            size = gym.spaces.flatdim(space)
            obs = flat_obs[:, index:index+size].view(batch_size, *space.shape)
            index += size
            return obs
        elif isinstance(space, gym.spaces.Discrete):
            size = gym.spaces.flatdim(space)
            obs = flat_obs[:, index:index+size]
            index += size
            return obs
        else:
            raise NotImplementedError(f"Space type {type(space)} not supported")
    
    unflat_obs = _unflatten(space)
    assert index == num_observations, "obs bigger than space"
    return unflat_obs


# 不需要嵌套函数的版本
# def unflatten(obs, space):
#     '''
#     obs: (num_envs, *shape)
#     space: unbatched observation space
#     '''
#     if isinstance(space, gym.spaces.Dict):
#         unflattened = {}
#         for key, subspace in space.spaces.items():
#             value, obs = unflatten(obs, subspace)
#             unflattened[key] = value
#         return unflattened, obs
#     elif isinstance(space, gym.spaces.Tuple):
#         unflattened = []
#         for subspace in space.spaces:
#             value, obs = unflatten(obs, subspace)
#             unflattened.append(value)
#         return unflattened, obs
#     elif isinstance(space, gym.spaces.Box):
#         size = gym.spaces.flatdim(space)
#         unflattened = obs[:, :size].view(-1, *space.shape)
#         remaining_obs = obs[:, size:]
#         return unflattened, remaining_obs
#     elif isinstance(space, gym.spaces.Discrete):
#         size = gym.spaces.flatdim(space)
#         unflattened = obs[:, :size]
#         remaining_obs = obs[:, size:]
#         return unflattened, remaining_obs
#     else:
#         raise NotImplementedError(f"Space type {type(space)} not supported")


def main():

    observation_space = gym.spaces.Dict(
        position=gym.spaces.Box(low=-1.0, high=1.0, shape=(3,)),
        image=gym.spaces.Box(low=0.0, high=1.0, shape=(3,64,64)),
        velocity=gym.spaces.Dict(
            x=gym.spaces.Box(low=-5.0, high=5.0, shape=(1,)),
            y=gym.spaces.Box(low=-5.0, high=5.0, shape=(1,)),
        ),
        status=gym.spaces.Discrete(4)
    )
    observation = torch.randint(low=0,high=10,size=(2, gym.spaces.flatdim(observation_space)))
    unflat_observation = unflatten_observation(observation, observation_space)
    flat_observation = FlattenObservation.flatten_observation(unflat_observation, observation_space)
    print(torch.all(observation == flat_observation))

    observation_space = gym.spaces.Dict(
        b=gym.spaces.Box(low=-1.0, high=1.0, shape=[1]),
        c=gym.spaces.Box(low=0.0, high=1.0, shape=[1]),
        a=gym.spaces.Dict(
            a=gym.spaces.Box(low=-5.0, high=5.0, shape=[1]),
            b=gym.spaces.Box(low=-5.0, high=5.0, shape=[1]),
        )
    )
    observation = {
        'c': torch.tensor([[30]]),
        'a': {
            'b': torch.tensor([[12]]),
            'a': torch.tensor([[11]])
        },
        'b': torch.tensor([[20]])
    }
    flat_observation = FlattenObservation.flatten_observation(observation, observation_space)
    unflat_observation = unflatten_observation(flat_observation, observation_space)
    print(observation)
    print(unflat_observation)


if __name__ == "__main__":
    main()