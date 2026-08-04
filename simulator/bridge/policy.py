"""Policy that hands control of the robot to an external process over websocket.

Thin wrapper over ``molmo_spaces.policy.learned_policy.websocket_policy
.WebsocketPolicy``, which already implements the msgpack-numpy protocol. All this
adds is a ``BasePolicyConfig`` so it can be dropped into an experiment config,
and a constructor matching the ``policy_factory(exp_config, task)`` signature the
data-generation pipeline calls.

See ``bridge/server.py`` for the other end.
"""

from __future__ import annotations

from typing import Any

from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.policy.base_policy import PolicyFactory
from molmo_spaces.policy.learned_policy.websocket_policy import WebsocketPolicy
from molmo_spaces.utils.function_utils import make_lenient


class BridgePolicy(WebsocketPolicy):
    """Connects to an external control server and applies whatever it returns."""

    def __init__(self, config, task: Any = None) -> None:
        remote = config.policy_config.remote_config
        super().__init__(
            config,
            model_name=config.policy_config.model_name,
            host=remote["host"],
            port=remote["port"],
            connection_timeout=config.policy_config.connection_timeout,
        )
        # WebsocketPolicy.__init__ does not forward `task`, but obs_to_model_input
        # needs it for the task description. The pipeline also sets this via
        # BaseMujocoTask.register_policy; assigning here keeps standalone use working.
        self.task = task

    def obs_to_model_input(self, obs):
        # Tolerate a task-less run (e.g. manual driving with no scripted task)
        # rather than crashing on `self.task.get_task_description()`.
        if self.task is None:
            obs = obs[0] if isinstance(obs, list) else obs
            return {**obs, "task": ""}
        return super().obs_to_model_input(obs)


class BridgePolicyConfig(BasePolicyConfig):
    """Experiment config selecting :class:`BridgePolicy`."""

    model_name: str = "bridge"
    remote_config: dict = dict(host="127.0.0.1", port=8000)
    connection_timeout: float | None = 120.0

    policy_cls: type = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "bridge"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            self.policy_cls = BridgePolicy
            self.policy_factory = make_lenient(BridgePolicy)
