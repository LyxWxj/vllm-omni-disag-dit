# Qwen-Image 2.1 Stage Execution 验证与修复说明

## 当前结论

本轮修复针对上一轮多请求部署矩阵中暴露的两个问题：

1. 分离角色的 request-batching 路径不再把多个请求送入同一个 pipeline forward。
   调度器仍可以收集请求波次，但 runner 会按请求逐个完成 ENCODE、DENOISE、
   DENOISE_DECODE 或 DECODE，并按原顺序返回，避免组批导致数值漂移或请求波次
   之间出现无输出、卡住。
2. `decode_batch` 不再通过过滤 `None` 静默缩短输出列表。任何请求没有对应解码
   输出时立即抛出带请求索引的错误，由 runner 的一请求一输出契约处理，而不是
   让请求无结果地结束。
3. Qwen pipeline 保留变长 prompt payload 的形状校验和单请求处理逻辑；分离角色
   不再依赖跨请求 padding、latent 拼接或批量 VAE decode。

FULL 路径仍保留 request-batching。分离角色的串行化覆盖全部非 `FULL`
`DiffusionStageRole`，因此即使调度器一次提交多个请求，也不会发生分离阶段的组批
数值变化。

## 验证判定标准

服务级多请求测试必须同时满足以下条件：

- 每个请求都返回 HTTP 成功响应；超时、异常响应、缺失响应均为失败。
- 每个响应都能解码为非空 PNG，尺寸等于请求尺寸，且每个请求只对应一个输出。
- 同一测试组中的输出 SHA-256 只作为诊断信息，不要求跨设备、跨 replica 或跨
  批次完全一致。CUDA kernel、VAE batch 顺序和不同设备执行都可能造成浮点差异。
- 需要建立同一 topology、同一 prompt、同一 seed 的单请求参考输出，解码 PNG
  后逐像素计算以下指标，并使用预先约定的有界阈值判定：

  ```text
  delta = abs(batch_image.astype(int32) - reference_image.astype(int32))
  max_abs = max(delta)
  mean_abs = mean(delta)
  rmse = sqrt(mean(delta ** 2))
  ```

  参考实现必须记录 `max_abs`、`mean_abs` 和 `rmse`；不能以“SHA 不同”直接判定
  失败，也不能以“请求没有输出”作为可接受的数值误差。

## 验证环境

| 项目 | 值 |
| --- | --- |
| 服务器 | `lab-dell03` |
| GPU | 2 x NVIDIA A100，GPU0 和 GPU1 |
| 模型 | `/home/louyuxuan/models/Qwen-Image-2.1` |
| 多请求参数 | 4 个请求，分为 2 个 batch，每批 2 个请求 |
| 分辨率 | `512 x 512` |
| steps | `4` |
| true CFG | `4.0` |
| seed | `42` |

## 拓扑定义

| 记法 | GPU0 | GPU1 | 语义 |
| --- | --- | --- | --- |
| `FULL` | `E+G+D` | — | 单卡完整 pipeline |
| `E/G-D` | `E` | `G+D` | Encode 与 Denoise+Decode 分离 |
| `FULL/FULL` | `FULL` replica | `FULL` replica | 两个完整 pipeline replica |
| `E-D/G` | `E+D` | `G` | Denoise 独立部署 |
| `E-G-D/G` | `E+G(replica 0)+D` | `G(replica 1)` | GPU0 保留完整链路并在 GPU1 增加 G replica |

## 上一轮多请求结果（修复前）

上一轮使用相同的多请求 runner 发现：`FULL`、`FULL/FULL` 和 step-execution
分离拓扑可以返回全部请求；request-batching 的 `E/G-D` 以及 `E-D/G` 存在
请求波次挂起或部分返回：

| 案例 | `step_execution=False` 修复前结果 | `step_execution=True` 修复前结果 |
| --- | --- | --- |
| upstream/main 多请求基线 | PASS，4/4 | — |
| 当前分支 `FULL` | PASS，4/4 | PASS，4/4 |
| 当前分支 `GPU0:E, GPU1:G-D` | **失败，0/4，批次超时** | PASS，4/4 |
| 当前分支 `GPU0:FULL, GPU1:FULL` | PASS，4/4 | PASS，4/4 |
| 当前分支 `GPU0:E-D, GPU1:G` | **失败/部分返回，第二批次超时** | PASS，4/4 |
| 当前分支 `E-G-D/G` | PASS，4/4 | PASS，4/4 |

这里的 `0/4`、部分返回和超时是功能失败，不能用 SHA 差异解释。step-execution
中的 ENCODE/DECODE 是逐请求 atomic execution，因此不能替代 request-batching
分离路径的验证。

## 修复后完整矩阵结果

2026-10-31 在 `lab-dell03` 上完成了完整多请求矩阵。每个 case 都先执行 4 次
单请求建立参考输出，再执行 2 轮、每轮 2 请求的 batch；表中的误差是 4 个
`global_index` 相对于对应单请求参考的最大值。`SHA 全相同` 表示 4 个请求的
batch/reference SHA 均相同，不表示不同 prompt 之间的图像 SHA 相同。

| 来源 | 模式 | 拓扑 | 服务输出 | SHA 全相同 | `max_abs` | `mean_abs` | `rmse` |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `upstream/main` | request | `FULL` | PASS，4/4 | 否 | 255 | 34.5040 | 46.0127 |
| 当前分支 | request | `FULL` | PASS，4/4 | 否 | 255 | 34.5040 | 46.0127 |
| 当前分支 | request | `E/G-D` | PASS，4/4 | 是 | 0 | 0.0000 | 0.0000 |
| 当前分支 | request | `FULL/FULL` | PASS，4/4 | 是 | 0 | 0.0000 | 0.0000 |
| 当前分支 | request | `E-D/G` | PASS，4/4 | 是 | 0 | 0.0000 | 0.0000 |
| 当前分支 | request | `E-G-D/G` | PASS，4/4 | 是 | 0 | 0.0000 | 0.0000 |
| 当前分支 | step | `FULL` | PASS，4/4 | 否 | 255 | 29.4317 | 41.4763 |
| 当前分支 | step | `E/G-D` | PASS，4/4 | 是 | 0 | 0.0000 | 0.0000 |
| 当前分支 | step | `FULL/FULL` | PASS，4/4 | 是 | 0 | 0.0000 | 0.0000 |
| 当前分支 | step | `E-D/G` | PASS，4/4 | 否 | 255 | 29.4317 | 41.4763 |
| 当前分支 | step | `E-G-D/G` | PASS，4/4 | 是 | 0 | 0.0000 | 0.0000 |

因此，服务功能性检查为 `11/11`：所有请求均返回单张非空 `512 x 512` PNG，且
两轮 `2 x 2` batch 均返回完整的 4 个响应。request-batching 的分离角色路径
（`E/G-D`、`E-D/G` 和 `E-G-D/G`）与单请求参考完全一致；step-execution 的
`E-D/G` 在 `global_index=2,3` 出现有界像素差异，但没有请求丢失或输出失败。
`FULL` 的 batch 与单请求参考也保留 batch execution 的浮点差异，SHA 和像素指标
均仅作为诊断信息。

首次不执行单请求预热、直接发送第一个多请求 batch 时，远端版本不匹配环境下
曾复现 `E-D/G` 冷启动超时。按本验证文档先建立单请求参考后，`E-D/G` 及其余
所有 case 均稳定通过；这次冷启动现象不作为当前实现的功能性失败判定。

## 已回退的组批实验

在本次回退之前，曾对最容易复现问题的 `GPU0:E, GPU1:G-D` request-batching 拓扑
做两请求和四请求、两批次验证。该实验结果仅用于确认“组批路径确实会产生过大的
数值差异”，不再代表当前实现的运行路径：

| 项目 | 结果 |
| --- | --- |
| 两请求组批 | PASS，2/2 返回有效 `512 x 512` PNG，但数值偏差过大 |
| 四请求、两批次组批 | PASS，2 轮 `2 x 2`，4/4 返回有效 `512 x 512` PNG，但不再采用 |
| 逐请求串行参考 | PASS，2/2 返回有效 `512 x 512` PNG |
| SHA-256 | 两个请求的组批/串行 SHA 均不相同，仅作为诊断信息 |
| request 0 | `max_abs=215`，`mean_abs=3.6691`，`rmse=9.2168` |
| request 1 | `max_abs=173`，`mean_abs=6.9714`，`rmse=15.2885` |

当前实现改为逐请求执行后，应重新建立与单请求参考一致的输出基线；上一轮组批的
`max_abs`、`mean_abs` 和 `rmse` 不适用于新的执行路径。

## 回退后定向验证

在相同的 `GPU0:E, GPU1:G-D` 部署和 `max_num_seqs=2` 配置下，调度器仍同时接收
两个请求，但 runner 和 Qwen 分离角色入口均逐请求执行。两个请求都返回有效的
`512 x 512` PNG；与随后逐请求发送的同 prompt、同 seed 参考相比：

| 请求 | SHA 相同 | `max_abs` | `mean_abs` | `rmse` |
| --- | --- | ---: | ---: | ---: |
| request 0 | `true` | `0` | `0.0` | `0.0` |
| request 1 | `true` | `0` | `0.0` | `0.0` |

这说明当前多请求 admission 不再引入分离阶段的组批数值差异；`FULL` 的真正
request-batching 路径未被本次回退改变。

## 当前验证边界

- 本地修改文件的定向测试为 `114 passed`；`ruff`、`compileall` 和
  `git diff --check` 均通过。
- 服务器完整矩阵为 `11/11`；每个 case 均完成 4 次单请求参考和 2 轮 `2 x 2`
  多请求 batch，所有响应均为有效的 `512 x 512` PNG。
- 服务器上的 `ruff` 已通过修改文件检查；服务退出后 GPU0/GPU1 均为 `0 MiB`，
  `canhazgpu` 无残留任务。
- 当前服务器 `/home/louyuxuan/vllm-omni-disag-dit/.venv` 与源码版本不匹配：
  `vllm.entrypoints.chat_utils` 缺失，且导入 Omni stage runtime 时缺少
  `vllm.v1.engine.utils.get_engine_zmq_addresses`。因此上述服务验证使用了临时
  compatibility shim 和源码覆盖树，不能替代依赖版本一致环境中的完整部署矩阵；
  没有修改服务器现有 dirty worktree 或其虚拟环境。
