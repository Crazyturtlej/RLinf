# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch

from rlinf.utils.logging import get_logger

logger = get_logger()

__all__ = ["RoboSimStudioEnv"]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Fetch config value for dict/attr config."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class RoboSimStudioEnv(gym.Env):
    """RLinf wrapper for RoboSimStudio environments.

    The wrapper:
    - Wraps RoboSimStudio's task-based Env interface
    - Exposes the canonical RLinf observation dict (``main_images``,
      ``wrist_images``, ``states``, ``task_descriptions``).
    - Supports metrics tracking, auto-reset and ``ignore_terminations``
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
        record_metrics: bool = True,
    ):
        """Initialize RoboSimStudio wrapper.

        Args:
          cfg: Config object/dict containing RoboSimStudio task settings.
          num_envs: Number of parallel envs.
          seed_offset: Seed offset added to cfg.seed.
          total_num_processes: Total num processes for interface parity.
          worker_info: Worker metadata.
          record_metrics: Whether to record episode metrics.
        """
        super().__init__()

        # Basic configuration
        env_seed = int(_cfg_get(cfg, "seed", 0))
        self.seed = env_seed + int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.cfg = cfg

        # RLinf standard settings
        self.auto_reset = bool(_cfg_get(cfg, "auto_reset", True))
        self.use_rel_reward = bool(_cfg_get(cfg, "use_rel_reward", False))
        self.ignore_terminations = bool(_cfg_get(cfg, "ignore_terminations", False))

        # Environment grouping
        self.group_size = int(_cfg_get(cfg, "group_size", 1))
        self.num_group = int(num_envs) // self.group_size
        self.use_fixed_reset_state_ids = bool(
            _cfg_get(cfg, "use_fixed_reset_state_ids", False)
        )

        self.video_cfg = _cfg_get(cfg, "video_cfg", None)
        self._device = torch.device("cpu")

        # RoboSimStudio specific settings
        self.task_name = str(_cfg_get(cfg, "task_name", "examples/crate_wash/flip"))
        self.max_steps = int(_cfg_get(cfg, "max_steps_per_rollout_epoch", 500))
        self.renderer = str(_cfg_get(cfg, "renderer", ""))
        self.render_every = int(_cfg_get(cfg, "render_every", 1))

        # Camera configuration
        self.main_camera = str(_cfg_get(cfg, "main_camera", "camera_0"))
        self.wrist_camera = str(_cfg_get(cfg, "wrist_camera", ""))

        # Image dimensions
        self.camera_height = int(_cfg_get(cfg.get("init_params", {}), "camera_heights", 256))
        self.camera_width = int(_cfg_get(cfg.get("init_params", {}), "camera_widths", 256))

        logger.info(f"Initializing RoboSimStudio environment with task: {self.task_name}")

        # Create individual RoboSimStudio environments
        self.envs = [self._make_env(i) for i in range(int(num_envs))]

        # Initialize observation and action spaces based on first env
        self._init_spaces()

        # Metrics tracking (ManiSkill/LIBERO style)
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.device
        )
        self.record_metrics = bool(record_metrics)
        self._is_start = True
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._needs_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self.info_logging_keys = ["success"]
        if self.record_metrics:
            self._init_metrics()

        self._last_obs: Optional[dict[str, Any]] = None
        self._last_info: dict[str, Any] = {}

    def _make_env(self, env_idx: int):
        """Create a single RoboSimStudio Env instance."""
        try:
            from robosimstudio_bench.envs import Env
        except ImportError as e:
            raise ImportError(
                "RoboSimStudio is not installed. Please install it with: "
                "pip install robosimstudio[bench]"
            ) from e

        # Create RoboSimStudio env with appropriate settings
        env = Env(
            task=self.task_name,
            episode=env_idx,  # Use env_idx for episode variation
            renderer=self.renderer if self.renderer else "",
            render_every=self.render_every,
            max_steps=self.max_steps,
            verbose=(env_idx == 0),  # Only verbose for first env
        )

        return env

    def _init_spaces(self):
        """Initialize observation and action spaces."""
        # Get a sample observation from first env
        sample_obs, _ = self.envs[0].reset()

        # Extract state dimension from proprio
        state_list = []
        if "proprio" in sample_obs:
            for key, value in sample_obs["proprio"].items():
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    state_list.append(np.array(value).flatten())

        self._state_dim = sum(s.size for s in state_list) if state_list else 0

        # Define observation space
        self.observation_space = gym.spaces.Dict({
            "states": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_envs, self._state_dim),
                dtype=np.float32,
            ),
            "main_images": gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.num_envs, self.camera_height, self.camera_width, 3),
                dtype=np.uint8,
            ),
        })

        # Action space - get actual dimension from RoboSimStudio env
        # This handles both single-arm (7D) and dual-arm (14D or 16D) tasks
        rss_action_space = self.envs[0].action_space
        if hasattr(rss_action_space, 'action_dim'):
            action_dim = rss_action_space.action_dim
        elif hasattr(rss_action_space, 'shape'):
            action_dim = rss_action_space.shape[0]
        else:
            # Fallback: infer from control mode
            action_dim = 7  # Default for single arm

        logger.info(f"Detected action dimension from RoboSimStudio: {action_dim}")

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

    # -------------------- Properties --------------------
    @property
    def total_num_group_envs(self) -> int:
        return np.iinfo(np.uint8).max // 2

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = bool(value)

    @property
    def instruction(self) -> list[str]:
        """Get task instructions for all environments."""
        instructions = []
        for env in self.envs:
            if hasattr(env, 'episode') and hasattr(env.episode, 'instruction'):
                instructions.append(env.episode.instruction)
            else:
                instructions.append("")
        return instructions

    # -------------------- Observation Conversion --------------------
    def _convert_obs_single(self, rss_obs: dict) -> dict[str, Any]:
        """Convert single RoboSimStudio observation to RLinf format.

        RoboSimStudio obs format:
        {
            "proprio": {"arm/joint_q": ..., "arm/gripper": ..., "arm/ee_pose": ...},
            "objects": {"obj_name": (px, py, pz, qx, qy, qz, qw), ...},
            "cameras": {"camera_name": {"rgb": ..., "depth": ..., "seg": ...}},
            "instruction": str,
            "step": int,
        }

        RLinf expected format (for OpenPi/VLA):
        {
            "states": torch.Tensor,  # Flattened proprioception
            "main_images": torch.Tensor,  # RGB image [H, W, 3]
            "wrist_images": torch.Tensor,  # Optional wrist view
        }
        """
        # Extract state (proprioception)
        state_list = []
        if "proprio" in rss_obs:
            for key in sorted(rss_obs["proprio"].keys()):
                value = rss_obs["proprio"][key]
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    state_list.append(np.array(value).flatten())

        states = np.concatenate(state_list) if state_list else np.zeros(self._state_dim)
        states = torch.from_numpy(states).float()

        # Extract images
        main_image = None
        wrist_image = None

        if "cameras" in rss_obs and rss_obs["cameras"] is not None:
            # Get main camera image
            if self.main_camera in rss_obs["cameras"]:
                cam_data = rss_obs["cameras"][self.main_camera]
                if "rgb" in cam_data:
                    main_image = np.array(cam_data["rgb"], dtype=np.uint8)

            # Get wrist camera if specified
            if self.wrist_camera and self.wrist_camera in rss_obs["cameras"]:
                cam_data = rss_obs["cameras"][self.wrist_camera]
                if "rgb" in cam_data:
                    wrist_image = np.array(cam_data["rgb"], dtype=np.uint8)

        # Create default images if not available
        if main_image is None:
            main_image = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        if wrist_image is None:
            wrist_image = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)

        main_image = torch.from_numpy(main_image)
        wrist_image = torch.from_numpy(wrist_image)

        return {
            "states": states,
            "main_images": main_image,
            "wrist_images": wrist_image,
        }

    def _collate_obs(self, obs_list: list[dict]) -> dict[str, Any]:
        """Collate list of observations into batched format."""
        out: dict[str, Any] = {}

        # Stack states
        states = torch.stack([o["states"] for o in obs_list], dim=0)
        out["states"] = states

        # Stack images
        main_images = torch.stack([o["main_images"] for o in obs_list], dim=0)
        out["main_images"] = main_images

        wrist_images = torch.stack([o["wrist_images"] for o in obs_list], dim=0)
        out["wrist_images"] = wrist_images

        # Add task descriptions
        out["task_descriptions"] = self.instruction

        return out

    def _collate_infos(self, info_list: list[dict]) -> dict[str, Any]:
        """Collate list of info dicts."""
        keys = set().union(*[inf.keys() for inf in info_list if isinstance(inf, dict)])
        out: dict[str, Any] = {}

        for k in sorted(keys):
            vals = [inf.get(k, None) for inf in info_list]
            is_bool = all(isinstance(v, (bool, np.bool_)) or v is None for v in vals)
            is_num = all(isinstance(v, (int, float, np.number)) or v is None for v in vals)

            if is_bool:
                out[k] = torch.tensor(
                    [bool(v) if v is not None else False for v in vals],
                    device=self.device,
                    dtype=torch.bool,
                )
            elif is_num:
                out[k] = torch.tensor(
                    [float(v) if v is not None else 0.0 for v in vals],
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                out[k] = vals

        return out

    # -------------------- Metrics --------------------
    def _calc_step_reward(self, reward: torch.Tensor) -> torch.Tensor:
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward
        return reward_diff if self.use_rel_reward else reward

    def _init_metrics(self) -> None:
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx: Optional[torch.Tensor] = None) -> None:
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self._elapsed_steps[mask] = 0
            if self.record_metrics:
                self.success_once[mask] = False
                self.returns[mask] = 0.0
        else:
            self.prev_step_reward[:] = 0.0
            self._elapsed_steps[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.returns[:] = 0.0

    def _record_metrics(
        self, step_reward: torch.Tensor, infos: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.record_metrics:
            return infos

        episode_info: dict[str, Any] = {}
        self.returns += step_reward

        if "success" in infos:
            self.success_once = self.success_once | infos["success"].bool()
            episode_info["success_once"] = self.success_once.clone()

        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        denom = torch.clamp(episode_info["episode_len"].float(), min=1.0)
        episode_info["reward"] = episode_info["return"] / denom
        infos["episode"] = episode_info

        return infos

    # -------------------- Core Environment Methods --------------------
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environments."""
        if options is None:
            options = {}

        env_idx = options.get("env_idx", None) if isinstance(options, dict) else None

        if env_idx is None:
            idxs = range(self.num_envs)
            self._reset_metrics()
            self._needs_reset[:] = False
        else:
            env_idx = torch.as_tensor(env_idx, dtype=torch.int64, device=self.device)
            idxs = env_idx.detach().cpu().tolist()
            self._reset_metrics(env_idx)
            self._needs_reset[env_idx] = False

        obs_list, info_list = [], []
        for i in range(self.num_envs):
            if i in idxs:
                # Reset RoboSimStudio env (it doesn't take seed param in reset)
                rss_obs, rss_info = self.envs[i].reset()
                obs_list.append(self._convert_obs_single(rss_obs))
                info_list.append(rss_info if isinstance(rss_info, dict) else {})
            else:
                # Use cached observation
                if self._last_obs is not None:
                    obs_list.append({
                        "states": self._last_obs["states"][i],
                        "main_images": self._last_obs["main_images"][i],
                        "wrist_images": self._last_obs["wrist_images"][i],
                    })
                else:
                    # Create dummy observation
                    obs_list.append({
                        "states": torch.zeros(self._state_dim),
                        "main_images": torch.zeros(self.camera_height, self.camera_width, 3, dtype=torch.uint8),
                        "wrist_images": torch.zeros(self.camera_height, self.camera_width, 3, dtype=torch.uint8),
                    })
                info_list.append({})

        obs = self._collate_obs(obs_list)
        infos = self._collate_infos(info_list)

        self._is_start = True
        self._last_obs, self._last_info = obs, infos
        return obs, infos

    def step(
        self,
        actions: Union[np.ndarray, torch.Tensor],
        auto_reset: bool = True,
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        """Step all environments."""
        # Normalize actions to numpy
        act_np = self._normalize_actions(actions)

        obs_list, info_list = [], []
        rew_list, term_list, trunc_list = [], [], []
        stepped_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for i, env in enumerate(self.envs):
            obs_i, info_i, rew_i, term_i, trunc_i, stepped = self._step_one_env(
                env_idx=i,
                env=env,
                action=act_np[i],
                auto_reset=auto_reset,
            )
            obs_list.append(obs_i)
            info_list.append(info_i)
            rew_list.append(rew_i)
            term_list.append(term_i)
            trunc_list.append(trunc_i)
            stepped_mask[i] = stepped

        self._elapsed_steps[stepped_mask] += 1

        obs = self._collate_obs(obs_list)
        infos = self._collate_infos(info_list)

        raw_reward = torch.tensor(rew_list, device=self.device, dtype=torch.float32)
        step_reward = self._calc_step_reward(raw_reward)

        terminations = torch.tensor(term_list, device=self.device, dtype=torch.bool)
        truncations = torch.tensor(trunc_list, device=self.device, dtype=torch.bool)

        infos = self._record_metrics(step_reward, infos)

        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics and "episode" in infos:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = bool(auto_reset) and bool(self.auto_reset)
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        self._last_obs, self._last_info = obs, infos
        return obs, step_reward, terminations, truncations, infos

    def _normalize_actions(
        self, actions: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """Normalize actions to numpy array."""
        act_np = (
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions)
        )
        if act_np.ndim == 1:
            act_np = np.repeat(act_np[None, :], self.num_envs, axis=0)
        if act_np.shape[0] != self.num_envs:
            raise ValueError(
                f"Invalid action batch dimension. Expected shape [num_envs, act_dim] "
                f"with num_envs={self.num_envs}, got {act_np.shape}."
            )
        return act_np.astype(np.float32, copy=False)

    def _step_one_env(
        self,
        env_idx: int,
        env: Any,
        action: np.ndarray,
        auto_reset: bool,
    ) -> tuple[dict[str, Any], dict, float, bool, bool, bool]:
        """Step a single RoboSimStudio environment."""
        if self._needs_reset[env_idx]:
            if auto_reset and self.auto_reset:
                env.reset()
                self._needs_reset[env_idx] = False
                self._reset_metrics(torch.tensor([env_idx], device=self.device))
            else:
                # Return cached obs with zero reward
                if self._last_obs is not None:
                    obs = {
                        "states": self._last_obs["states"][env_idx],
                        "main_images": self._last_obs["main_images"][env_idx],
                        "wrist_images": self._last_obs["wrist_images"][env_idx],
                    }
                else:
                    obs = {
                        "states": torch.zeros(self._state_dim),
                        "main_images": torch.zeros(self.camera_height, self.camera_width, 3, dtype=torch.uint8),
                        "wrist_images": torch.zeros(self.camera_height, self.camera_width, 3, dtype=torch.uint8),
                    }
                return obs, {}, 0.0, True, False, False

        # Step RoboSimStudio env
        # RoboSimStudio returns: obs, reward, terminated, truncated, info
        rss_obs, rew, terminated, truncated, info = env.step(action)

        obs = self._convert_obs_single(rss_obs)
        info = info if isinstance(info, dict) else {}

        return obs, info, float(rew), bool(terminated), bool(truncated), True

    def _handle_auto_reset(
        self, dones: torch.Tensor, obs: dict[str, Any], infos: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Handle auto-reset for done environments."""
        done_ids = torch.where(dones)[0]

        if len(done_ids) == 0:
            return obs, infos

        # Reset done environments
        reset_obs, reset_infos = self.reset(options={"env_idx": done_ids})

        # Update observations for reset environments
        for i, done_id in enumerate(done_ids):
            obs["states"][done_id] = reset_obs["states"][i]
            obs["main_images"][done_id] = reset_obs["main_images"][i]
            obs["wrist_images"][done_id] = reset_obs["wrist_images"][i]

        return obs, infos

    def close(self):
        """Close all environments."""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()
