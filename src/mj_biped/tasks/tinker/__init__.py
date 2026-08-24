from mjlab.rl import RslRlModelCfg
import torch

from .tinker_env_cfg import tinker_env_cfg
from .mdp.runner import tinker_ppo_runner_cfg

from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Mjlab-Tinker",
  env_cfg=tinker_env_cfg(),
  play_env_cfg=tinker_env_cfg(play=True),
  rl_cfg=tinker_ppo_runner_cfg(),
)
# 1. Load your model (replace MyModel with your actual class/architecture)
model = RslRlModelCfg()
model.load_state_dict(torch.load("model_3500.pt"))
model.eval()

# 2. Create a dummy input with the correct shape (e.g., batch size 1, 3 channels, 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Export the model
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
)