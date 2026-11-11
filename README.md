# Atomic-PPO O0

O0 是 Atomic-PPO 双车型车队派单与调度的完整 OD 基准。车队观察使用按车型、车辆状态、区域和 ETA 分组的计数，ETA 桶为 0–7 分钟及 8 分钟以上。Actor 和 Critic 均使用按订单类型区分的完整起讫点（OD）输入。Actor 使用按物理分钟计算的 GAE，Critic 使用以平均收益为中心的完整回合回报。

- `experiment_l_histogram_eta.py` 是训练和验证入口，`--experiment od --od-group O0` 选择 O0。
- `histogram_eta.py` 定义观察表示、网络和采样适配，`physical_time_gae.py` 实现分钟 GAE。
- `atomic_dual/` 提供车队仿真、PPO 更新、轨迹采样、数据读取和评估。
- `tests/test_histogram_eta.py` 用合成场景检查模型、训练、评估和恢复。

## 安装和测试

在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
python -m pytest -q tests/test_histogram_eta.py
```

CUDA 相关测试需要支持 CUDA 的 PyTorch 和可用的 NVIDIA GPU。CPU 环境会跳过这些测试。

## 运行 O0

公开仓库提供源码和合成测试。复现原实验需要自行准备以下输入，并保持场景文件顺序及内容与参考运行一致：

| 命令参数 | 需要的文件 |
| --- | --- |
| `--run` | 参考运行的 `manifest.json` 和 `validation_scenarios/*.npz` |
| 参考运行的 `manifest.json` 中的 `arguments.data` | 该目录下的 `evening.npz`。相对路径以当前工作目录为准。 |
| `--snapshots` | 快照目录下的 `manifest.json` 和 `trusted_local_snapshots.pt` |
| `--bank` | 动作库目录下的 `action_probabilities.npz` |

即使设置 `--q-replicates 0`，训练入口也会加载快照和动作库。只使用可信来源的 `trusted_local_snapshots.pt`，该文件包含序列化的仿真对象。

以下命令使用 seed 9 实验的训练参数。将三个输入目录替换为你的实际路径：

```powershell
python experiment_l_histogram_eta.py --experiment od --od-group O0 `
  --run ./inputs/reference --snapshots ./inputs/snapshots --bank ./inputs/bank `
  --output ./runs/O0_seed9 --seeds 9 --iterations 300 --episodes 30 --workers 1 `
  --device cuda --rollout-backend batched --minute-lambda 0.95 `
  --validation-algorithm ppo_sampled --eval-every 300 --q-replicates 0
```

此配置使用 CPU 批量采样和 CUDA PPO 更新。可加 `--dry-run` 检查输入与模型初始化。输出目录包含运行清单以及 `O0_seed9/` 下的训练日志、验证记录和检查点。
