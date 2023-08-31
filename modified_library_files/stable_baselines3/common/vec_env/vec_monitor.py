import time
import warnings
from typing import Optional, Tuple

import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper


class VecMonitor(VecEnvWrapper):
    """
    A vectorized monitor wrapper for *vectorized* Gym environments,
    it is used to record the episode reward, length, time and other data.

    Some environments like `openai/procgen <https://github.com/openai/procgen>`_
    or `gym3 <https://github.com/openai/gym3>`_ directly initialize the
    vectorized environments, without giving us a chance to use the ``Monitor``
    wrapper. So this class simply does the job of the ``Monitor`` wrapper on
    a vectorized level.

    :param venv: The vectorized environment
    :param filename: the location to save a log file, can be None for no log
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    def __init__(
        self,
        venv: VecEnv,
        filename: Optional[str] = None,
        info_keywords: Tuple[str, ...] = (),
    ):
        # Avoid circular import
        from stable_baselines3.common.monitor import Monitor, ResultsWriter

        # This check is not valid for special `VecEnv`
        # like the ones created by Procgen, that does follow completely
        # the `VecEnv` interface
        try:
            is_wrapped_with_monitor = venv.env_is_wrapped(Monitor)[0]
        except AttributeError:
            is_wrapped_with_monitor = False

        if is_wrapped_with_monitor:
            warnings.warn(
                "The environment is already wrapped with a `Monitor` wrapper"
                "but you are wrapping it with a `VecMonitor` wrapper, the `Monitor` statistics will be"
                "overwritten by the `VecMonitor` ones.",
                UserWarning,
            )

        VecEnvWrapper.__init__(self, venv)
        self.episode_returns = None
        self.episode_social_welfare = None
        self.episode_lengths = None
        self.episode_apples_consumed = None
        self.episode_beams_fired = None
        self.episode_times_hit = None
        self.episode_tiles_cleaned = None
        self.episode_count = 0
        self.t_start = time.time()

        env_id = None
        if hasattr(venv, "spec") and venv.spec is not None:
            env_id = venv.spec.id

        if filename:
            self.results_writer = ResultsWriter(
                filename, header={"t_start": self.t_start, "env_id": env_id}, extra_keys=info_keywords
            )
        else:
            self.results_writer = None
        self.info_keywords = info_keywords

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_social_welfare = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_apples_consumed = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_beams_fired = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_times_hit = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_tiles_cleaned = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_social_welfare += np.asarray([i.get('social_welfare') for i in infos])
        self.episode_apples_consumed += np.asarray([i.get('apples_consumed') for i in infos])
        self.episode_beams_fired += np.asarray([i.get('beams_fired') for i in infos])
        self.episode_times_hit += np.asarray([i.get('times_hit') for i in infos])
        self.episode_tiles_cleaned += np.asarray([i.get('tiles_cleaned') for i in infos])
        self.episode_lengths += 1

        new_infos = list(infos[:])
        #print('vec_monitor_infos', infos)
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_social_welfare = self.episode_social_welfare[i]
                episode_apples_consumed = self.episode_apples_consumed[i]
                episode_beams_fired = self.episode_beams_fired[i]
                episode_times_hit = self.episode_times_hit[i]
                episode_tiles_cleaned = self.episode_tiles_cleaned[i]

                episode_length = self.episode_lengths[i]

                #episode_info = {"r": episode_return, "l": episode_length, "t": round(time.time() - self.t_start, 6)}
                episode_info = {"r": episode_return, "l": episode_length, "t": round(time.time() - self.t_start, 6), "social_welfare": episode_social_welfare, "apples_consumed": episode_apples_consumed, "beams_fired": episode_beams_fired, "times_hit": episode_times_hit, "tiles_cleaned": episode_tiles_cleaned}
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                self.episode_social_welfare[i] = 0
                self.episode_apples_consumed[i] = 0
                self.episode_beams_fired[i] = 0
                self.episode_times_hit[i] = 0
                self.episode_tiles_cleaned[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
        return obs, rewards, dones, new_infos

    def close(self) -> None:
        if self.results_writer:
            self.results_writer.close()
        return self.venv.close()
