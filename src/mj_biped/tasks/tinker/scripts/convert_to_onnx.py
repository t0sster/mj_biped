import torch
from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

CHECKPOINT_PATH = "logs/rsl_rl/tinker/2026-08-25_14-19-03/model_2100.pt"
TASK_ID = "Mjlab-Tinker"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "/home/ilya/be2r/Biped/politics/tinker/mj_biped/src/mj_biped/tasks/tinker/exported/"
OUTPUT_FILENAME = "policy.onnx"

def export_onnx():
    configure_torch_backends()

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=DEVICE)
    runner.load(
        CHECKPOINT_PATH,
        load_cfg={"actor": True},
        strict=True,
        map_location=DEVICE,
    )

    runner.export_policy_to_onnx(OUTPUT_DIR, filename=OUTPUT_FILENAME)
    print(f"Модель успешно экспортирована в {OUTPUT_DIR}/{OUTPUT_FILENAME}")


if __name__ == "__main__":
    export_onnx()
