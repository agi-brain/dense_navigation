from xuance.environment.vector_envs.vector_env import VecEnv, AlreadySteppingError, NotSteppingError
from xuance.common import space2shape, combined_shape
from gym.spaces import Dict
import numpy as np
from xuance.environment.gym.gym_vec_env import SubprocVecEnv_Gym, DummyVecEnv_Gym, worker


class SubprocVecEnv_Drones(SubprocVecEnv_Gym):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, context='spawn', in_series=1):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        super(SubprocVecEnv_Drones, self).__init__(env_fns, context, in_series)


class DummyVecEnv_Drones(DummyVecEnv_Gym):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.obs_shape = space2shape(self.observation_space)
        if isinstance(self.observation_space, Dict):
            self.buf_obs = {k: np.zeros(combined_shape(self.num_envs, v)) for k, v in
                            zip(self.obs_shape.keys(), self.obs_shape.values())}
        else:
            self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool_)
        self.buf_trunctions = np.zeros((self.num_envs,), dtype=np.bool_)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        try:
            self.max_episode_length = env.max_episode_steps
        except AttributeError:
            self.max_episode_length = 1000
