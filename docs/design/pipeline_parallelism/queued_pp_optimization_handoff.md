# Queued PP 优化交接

更新时间：2026-10-07

当前分支：`feature/pipeline_parallelism`

当前已签名提交：`b77d9def8bec8b9d64c4c93b70dcb7e4ae739d34` (`Restore queued PP throughput overlap`)

## 当前状态

当前实现以 Wan2.2、PP2、step execution 和 queued pipeline 为目标。每个请求在 Engine 中拥有一个 queued pipeline batch，两个 PP stage 通过 activation edge 和 feedback edge 传递中间张量和 scheduler 更新。普通静态 PP 仍走原有 pipeline 路径，queued PP 的控制状态集中在 queued batch 生命周期中。

已经完成的主要工作如下：

- 把 StageEngine 改成 Worker 内部的单 owner 进度线程，负责串行执行 Worker RPC、stage forward 和 transport progress。
- 为 activation 和 feedback 建立有界 connector、send/receive lease、transfer credit 和 identity 校验。
- 将 queued batch 生命周期固定为 prepare、submit、authorize、step complete、finalize、retire，并保留取消和失败清理路径。
- 将 final VAE decode 放到独立的 `WanFinalDecode` 线程和 CUDA stream 中，使 final decode 可以与后续 DiT forward 重叠。
- 保留 VAE1 的 output-owner 负载均衡；启用 `vae_patch_parallel_size>1` 时，两个 PP rank 参加同一个分布式 VAE decode。
- 保留 ContextVar forward context、VAE finalization quiescence 和 VAE group 隔离，避免 finalization 与 PP collective 乱序。

## 最新 trace 结论

最新 trace 使用 4 请求、1024×1024、80 帧、8 steps、`VLLM_OMNI_PP_TRACE_SYNC=1`，并且没有在命令行显式传入 `edge_buffer_slots`。运行代码的默认值为 2，4/4 请求成功完成。

可视化文件位于：

`artifacts/wan22-pp-stage-trace-4x8-edge2-sync-20261007/wan22_pp_stage_timeline.html`

原始 rank trace、manifest 和汇总 JSON 位于同一目录。trace 只展示 feature queued PP2 的 stage0/stage1 时间轴，包含 TextEncoder、DiT forward 和 VAE Decoder。

### VAE 时间区间的解释

VAE 区间不是普通 PP stage forward 的一部分。最后一个 denoise step 完成后，Engine 选择 finalization output owner，Worker 将 `post_decode()` 提交到 `WanFinalDecode` 线程；该线程在独立 CUDA stream 上调用 `vae.decode()`。因此，VAE decode 与 StageEngine 继续执行的 DiT forward 可以在同一个 rank 上重叠，这是当前设计主动保留的并发。

VAE1 模式下，Engine 根据各 rank 当前 outstanding finalization 数量在两个 PP rank 之间轮换 output owner，所以两个 stage 都可能出现 VAE 区间。VAE2 模式下，两个 rank 会共同执行分布式 VAE decode，因此两个 rank 同时出现 VAE 区间也是预期行为。

当前 trace 的区间边界基本正确，但标注信息还不完整：trace 记录了 PP rank，没有记录 finalization batch、线程名和 output owner；首个短 VAE 区间还可能来自 warmup。因此可视化中的 `stage0/stage1 · VAE Decoder` 应理解为“该 PP rank 上发生的 VAE 调用”，不能理解为“该 rank 的 DiT stage 正在执行 VAE”。后续若继续维护 trace，应在事件中增加 `thread_name`、`thread_id`、`batch_id`、`request_ids`、`output_owner` 和 `phase`。

### DiT 空泡的根因和修复

之前 `edge_buffer_slots=1` 时，每条 activation/feedback edge 只有一个在途 credit。一个 stage 完成 forward 后，下一次 forward 必须等对端消费 activation、释放 receive lease 和归还 send credit；这会产生接近一次 DiT forward 时长的停顿。

4 请求、8 steps、同步 trace 的 A/B 结果如下：

| edge buffer slots | 总耗时 | 吞吐 | 数值结果 |
| ---: | ---: | ---: | --- |
| 1 | 64.50s | 0.06202 req/s | 基线 |
| 2 | 55.88s | 0.07158 req/s | 4 个输出 hash 完全一致 |

因此将 queued PP 的默认 edge buffer 从 1 调整为 2。它只解决单 credit 带来的第一层 backpressure，不等于已经消除了所有 queued PP 空泡：

- `vllm_omni/diffusion/data.py`
- `vllm_omni/config/omni_config.py`
- `tests/diffusion/test_queued_pipeline_config.py`

显式传入 `edge_buffer_slots=1` 仍然会选择保守的单 credit 行为；需要观察连续 stage execution 时，应使用 2。增大该值会增加一个 activation buffer 的显存占用，后续清理不能删除 stage buffer budget 和 lease 检查。

最新 edge2 trace 仍能看到约 1s 级别的间隙。关闭设备同步后，类似间隙仍然存在，因此它们不是 `VLLM_OMNI_PP_TRACE_SYNC=1` 单独造成的。worker 事件显示，上一批 feedback 完成、下一批被 Engine 接受和授权之后，stage forward 还要等待下一轮 autonomous StageEngine/transfer grant；这属于 step commit、授权和 transfer control 的串行化开销。设备同步主要增加 span 前后的等待和约十几毫秒级的小间隙，也会扰动总耗时，但不会解释这些完整 forward 长度的空泡。

后续性能清理需要把这两类问题分开：保留 edge2 作为安全的最小 credit 配置，同时单独梳理“STEP_COMPLETED → scheduler commit → submit/authorize next step → transfer grant”的控制链。只有在不改变 request ownership、取消、collective 顺序和数值结果的前提下，才能合并这些控制 round；删除轮询或 lease 检查本身不能作为空泡修复。

专项控制 trace 已保存在 `artifacts/wan22-grant2-control-trace-4x8/`，记录了 Engine 的 scheduler commit/authorize、Worker 的 StageEngine tick、grant start 和 stage progress。进一步增加 StageEngine 命令实际执行时间的专项日志后，详细结果保存在 `artifacts/wan22-control-detail-4x8/`：`authorize_pipeline_batch()` 的 Worker 执行体只有约 `0.1–0.5ms`，`start_pipeline_transfer()` 约 `0.2–3.5ms`，但 Engine 侧一次 `authorize()` 仍可能耗时约 `0.97–0.99s`。

慢 `authorize()` 的原因已经可以由时间戳闭环确认：控制 RPC 到达时，某个 Worker 的 StageEngine owner 正在执行约 `0.975–0.995s` 的 `pipeline_stage_engine_tick`，后续 `enqueue_pipeline_batch` 和 `authorize_pipeline_batch` 只能在该 tick 结束后才真正开始执行。因此这里是“控制调用被计算占用的 StageEngine 串行化”，不是 `authorize` 判断逻辑本身很慢，也不是 `start_pipeline_transfer` 的数据传输耗时。Stage1 的 tick 仍然必须等待对应 activation 到达，这是必要的数据依赖；但等待 activation 之后，scheduler/control RPC 与下一轮 StageEngine tick 仍共用 owner，造成了可消除的控制空泡。

尝试将 autonomous multiprocess executor 的 `grant_limit` 从 1 提到 2 后，控制 trace 中 activation/feedback 仍没有稳定地在同一轮 ready。关闭 trace 后，同一负载的 grant limit1 为 `56.12s`，grant limit2 为 `58.61s`，没有可复现收益，所以该尝试已回滚。下一步应优先研究如何让“已完成的 StageEngine tick 的结果发布、下一批授权和 transport grant”不再等待同一个 owner 的同步 RPC；同时保留 activation 到达、FIFO、consumer device event 和 lease 检查。不应继续单纯提高 grant limit，也不应把 transfer start 从 owner 线程移出而不先解决 P2P 操作顺序和 tensor lifetime。

## 异步 admission/control lane 设计

当前 autonomous 路径的关键串行点是：Engine 通过同步 `collective_rpc()` 调用 `enqueue_pipeline_batch` 和 `authorize_pipeline_batch`，Worker 再通过 `PipelineStageEngine.call()` 等待 owner 线程执行。若 owner 此时正在运行一个约 1 秒的 DiT tick，控制 RPC 就会被计算占住。目标是把“控制命令入队”和“控制命令执行完成”拆开，而不是把模型计算或 P2P 操作搬到另一个线程。

建议按以下协议实现第一版：

1. 新增一个 queued PP 专用的 `admit_pipeline_batches` 命令，将同一轮可授权的 `enqueue + authorize` 合并成一次 StageEngine command。请求仍由 Worker busy loop 在两张卡上按同一 collective 顺序接收，但通过 `StageEngine.submit()` 入队后立即返回 `queued` ack。
2. StageEngine 继续是唯一修改 `PipelineStageState`、调用模型和启动 P2P 的 owner。当前 tick 结束后，它按队列顺序执行 admission command；因此不会引入 CUDA stream、NCCL 顺序或 tensor ownership 的新并发关系。
3. Worker 完成 command 后通过现有 `PipelineWorkerUpdate` 通道发布 admission ack/error。Executor 只有在 stage0 和 stage1 都报告同一个 `(batch_id, epoch)` 的 ack 后，才向 Engine 暴露 `SUBMITTED`/`AUTHORIZED` 状态。不能因为 command 已入队就提前把 batch 当成可运行。
4. Engine 增加 `ADMISSION_PENDING` 状态，并让 admission capacity 同时计入 `ADMISSION_PENDING`、`SUBMITTED` 和 `AUTHORIZED`，防止异步提交期间重复占用 `max_inflight_batches`。`_advance_queued_pipeline_batch()` 在 pending 时只保留 ownership，收到双 stage authorization ack 后再进入正常 progress。
5. 一旦收到任一 stage 的 command error、队列溢出或身份不匹配，Executor 立即让整个 queued control lane 失败；不允许另一张卡继续消费孤立 admission。取消和 shutdown 必须保持在已提交 command 之后的顺序，并清理未完成 ack。

第一版应只作用于 `_uses_autonomous_pipeline_stages()`，普通 PP 和旧的同步路径保持原协议。实现顺序建议是：先增加 admission ack 数据结构和状态机单测，再接入单批异步 command，随后合并同一 scheduler round 的多个 batch，最后用 4 请求和 32 请求的同构/异构负载验证吞吐、bubble、输出 hash 和 `1e-8` 数值容差。每一步都要保留同步路径作为回退开关，便于区分 admission 优化收益和调度回归。

本轮 merge commit `1893402244d37dbdd90211cdb3e74e0d7c04c67e` 在服务器上完成了 correctness、精度和性能回归。manifest 保存在 `artifacts/wan22-merge-20261008/`：

- 4 请求 no-sync accuracy：`accuracy-4x8-manifest.json`，`1024×1024`、80 帧、8 steps、PP2、edge2，4/4 成功，耗时 `53.17s`，吞吐 `0.07523 req/s`。
- 32 请求同构、无同步：`homo32-manifest.json`，`1024×1024`、80 帧、8 steps，32/32 成功，耗时 `442.56s`，吞吐 `0.07231 req/s`。
- 32 请求异构、无同步：`hetero32-manifest.json`，profile 按 `1024×1024`、`512×512`、`1280×704`、`768×768` 循环，32/32 成功，耗时 `326.15s`，吞吐 `0.09811 req/s`。

精度对照保存在 `accuracy-high-manifest.json` 对应的 paired run：merge 与 c50 基线在两个 seed、`1024×1024×80×8` 上 decoded tensor 逐元素 `max_abs=0.0`、`mean_abs=0.0`；小尺寸两请求 hash 也完全一致。异构吞吐不能直接与同构吞吐作等价结论，因为四种 profile 的平均像素量更低；比较时应按像素量、帧数和 steps 归一化，或分别报告各 profile 的延迟。

最终 feature/main 对比使用相同的 32 请求、80 帧、8 steps 和四种异构 profile；feature 数值为本轮新 benchmark，main 数值沿用现有静态 PP2 对照基线。feature 使用 queued PP2、`edge_buffer_slots=2`、`vae_patch_parallel_size=1`，main 使用静态 PP2、`step_execution=false`、`vae_patch_parallel_size=2`：

| 分支和负载 | 同步 | 总耗时 | 吞吐 | 完成数 |
| --- | --- | ---: | ---: | ---: |
| feature queued PP2，同构 1024×1024 | 关闭 | 442.56s | 0.07231 req/s | 32/32 |
| feature queued PP2，异构 | 关闭 | 326.15s | 0.09811 req/s | 32/32 |
| main 静态 PP2，同构 1024×1024 | 关闭 | 818.07s | 0.03912 req/s | 32/32 |
| main 静态 PP2，异构 | 关闭 | 525.64s | 0.06088 req/s | 32/32 |

异构 profile 固定为 `1024×1024`、`512×512`、`1280×704`、`768×768` 循环排列。按现有 main 对照基线计算，feature 相对 main 的吞吐约为同构 `1.85x`、异构 `1.61x`；异构组的绝对吞吐仍受平均像素量较低影响，不能直接等价为相同计算量加速。feature manifest 保存在 `artifacts/wan22-merge-20261008/`，main manifest 沿用 `artifacts/wan22-final-main-homo32/` 和 `artifacts/wan22-final-main-hetero32/`。

### 测试方法与同步

- **本地静态检查：** 激活仓库 `.venv` 后运行 `ruff check`、`python -m compileall` 和 `git diff --check`。本地 pytest 需要完整且可读的 `.venv` 依赖；如果 `transformers/models/rembert` 等环境文件触发 `Errno 5`，应记录为环境阻断，不修改依赖目录来掩盖问题。
- **远程功能/性能：** 目标机器为 `lab-dell03`，仓库为 `~/vllm-omni-disag-dit`，使用 `.venv/bin/python`,使用`canhazgpu`来提交任务。feature 测试前确认 `git rev-parse HEAD` 与待测 commit 一致；运行 Wan2.2 PP2 后保存 `manifest.json`、请求 hash 和必要的 worker events。
- **trace：** 只有需要分析时间线时设置 `VLLM_OMNI_PP_TRACE_DIR` 和 `VLLM_OMNI_PP_TRACE_SYNC=1`；性能 benchmark 必须取消这两个变量，避免同步和写日志改变调度。`1280×720` profile 直接写成 `1280×704`。
- **main 对照：** `git checkout main && git pull --ff-only origin main`，main 静态 PP2 使用 `step_execution=false`、`vae_patch_parallel_size=2`；feature queued PP2 使用 `step_execution=true`、`vae_patch_parallel_size=1`、`edge_buffer_slots=2`。两者共享同一请求数量、分辨率、帧数、steps 和 seed。
- **本地与远程同步：** 本地 feature 完成检查后使用带签名的 `git commit -S`，推送 `git push origin feature/pipeline_parallelism`；服务器先确认工作树干净，再运行 `git checkout feature/pipeline_parallelism && git pull --ff-only origin feature/pipeline_parallelism`。main 对照结束后恢复目标分支，并确认 `git status --short --branch`。

## 空泡归因

后续清理必须保留下面几类边界，不能把它们统称为“多余轮询”或“可以删除的等待”：

1. **流水线热身和排空。** 第一批 activation 必须经过 stage0 → transfer → stage1，最后一批还要完成 feedback 和 VAE finalization。这些边界属于流水线本身，不能用 steady-state bubble 的标准判断。
2. **StageEngine owner 串行化。** stage0 完成 forward 后虽然已经产生 activation offer，但 grant/start 命令仍要通过同一个 StageEngine owner。如果 owner 立即执行下一次 forward，`start_pipeline_transfer()` 会排在这次计算之后，stage1 只能等待 transfer 真正启动。控制命令短不代表等待短，等待发生在 command queue 前面。
3. **active grant 和 consumer lease。** `grant_ready()` 会让两个 endpoint 保持 busy，直到 send completion、receive lease 和 device consumer event 都完成。过早释放会导致 tensor lifetime、NCCL 顺序或重复 grant 错误；它造成的等待属于真实 credit/lease 依赖。
4. **in-flight capacity 和 feedback 依赖。** 当 `max_inflight_batches=2` 时，两个 batch 占满 admission 后，stage0 可能必须等 stage1 完成 activation 并返回 feedback，才能 commit 当前 batch、授权下一个 batch。这个空档是数据依赖和 capacity 的组合，增加 in-flight 数量会改变它，同时增加显存占用。
5. **pending receive 的唤醒条件。** receive message 可能已经从 transport 进入 pending 队列，但当时尚未满足 FIFO 或 authorization 条件；如果 `pipeline_stage_engine_needs_progress()` 把它当成无工作，StageEngine 会睡到下一次外部 control wake，形成约一个 forward 周期的空泡。
6. **未启动 send 的唤醒条件。** send ticket 已经保留但 grant 尚未 start 时，transport 本身可能还没有 outstanding backend operation。若 StageEngine 此时停止 polling，grant start 和下一次计算之间会形成串行 round；stage0 也可能绕过 gate 继续取下一个任务。
7. **设备同步的测量扰动。** `VLLM_OMNI_PP_TRACE_SYNC=1` 会在 span 边界插入 device synchronize，增加毫秒级边界等待并改变调度时序。此前 no-sync trace 也出现完整 forward 长度的空泡，因此同步不是这些大空泡的根因；性能 benchmark 仍必须关闭同步。
8. **trace 时间戳口径。** worker hook 中的 `stage_progress.timestamp_ns` 是进入 `progress_pipeline()` 前记录的时间，不能当作 forward 结束时间；forward 起止必须以 PP trace 的 `dit.forward` `B/E` span 为准，控制事件再用 worker event 对齐。

## 最新性能验证

本轮 4 请求同步 trace 保存在 `artifacts/wan22-queued-pp-cleanup-20261007/trace/`，用于确认 steady-state 空泡归因；同构和异构 no-sync manifest 用于吞吐、完成数和 hash 回归。

## 尚未完成的清理任务

下一阶段的目标是删除历史遗留的冗余测试和功能代码，保持数值精度、吞吐和现有公共接口。清理应该围绕“集中实现、薄公共适配层”进行，不继续把 queued 专用逻辑散落到 `diffusion_engine.py`、`diffusion_worker.py`、`multiproc_executor.py` 和 `uniproc_executor.py`。

最新性能修复已经改变了清理优先级：先冻结 admission/control lane 和 StageEngine progress gate 的行为，再做集中重构。以下逻辑属于性能正确性边界，清理时必须保留对应回归测试：

- `ADMISSION_PENDING`、双 stage ack 和 capacity accounting，不能提前把未执行的 admission 当成可运行 batch。
- `PipelineStageEngine` 的 command queue 顺序、未启动 send 的 polling，以及 stage0 在 activation grant/start 前的 forward gate。
- pending receive message 的持续 polling、consumer lease、device event 和 transfer retirement 顺序。
- `start_pipeline_transfer` 与 DiT forward 的 owner 顺序，以及异步 admission command 的错误传播和取消顺序。

清理应先删除重复的包装、旧 benchmark runner 和重复事件解析，再把 queued 状态机和 Worker runtime 集中到新模块；每次移动逻辑都要保留 4 请求同步 trace、32 请求同构/异构 benchmark 和数值 hash 回归。不能在性能未回归前把上述条件判断合并成一个笼统的“有 pending work”判断。

当前清理进度：

- 已将 Engine queued batch phase、batch ownership、finalization quiescence 和 output-owner 选择移入 `vllm_omni/diffusion/queued_pp/runtime.py`；`DiffusionEngine` 保留兼容委托。
- 本阶段新增 `vllm_omni/diffusion/queued_pp/executor_adapter.py`，统一 multiprocess/uniprocess Executor 对 rank 聚合 list envelope 的展开；readiness 类型校验、rank coverage、StageEngine 顺序和 transport 生命周期仍在各自实现中。
- 本阶段删除两套 Executor 中重复的 result-envelope 代码，未修改普通 execute/static PP RPC。远端 Executor/connector focused suite 为 `84 passed`，Engine/retirement suite 为 `63 passed`。
- 合并后的完整 queued correctness focused suite 共 `302 passed`，包含 upstream 新增的 VAE batch/recovery 测试；下一阶段可以继续 Worker runtime 抽取。
- 本阶段新增 `vllm_omni/diffusion/queued_pp/worker_runtime.py`，集中 Worker finalization future、executor、published 状态、device event 和 CUDA stream；Worker 保留兼容属性与原有 model/StageEngine 调用顺序。Worker/Engine/retirement focused suite 为 `129 passed`。
- upstream merge 后 stage0 activation grant/start gate 的旧测试期望已对齐为 gate contract：未启动前不会继续发第二个 activation，避免把真实 backpressure 当成回归。

### 1. 建立 queued PP 专用模块边界

建议新增一个集中目录：`vllm_omni/diffusion/queued_pp/`。第一步可以只引入一个 `runtime.py`，把以下内容集中起来：

- `QueuedPipelineBatch`、phase 枚举和 batch ownership 检查。
- admission limit、prepared/submitted/authorized batch 计数。
- step complete、scheduler commit、finalization pending/finalizing、retire 的状态迁移。
- output owner 选择和 finalization quiescence 判断。
- 与 executor 交互所需的窄接口协议。

`DiffusionEngine` 只保留请求调度循环和少量委托调用，例如 `prepare()`、`advance()`、`finalize_ready()` 和 `retire_ready()`。Engine 中重复出现的 `progress_pipeline()`、`poll_pipeline_events()`、release readiness 和 transfer retirement 判断应逐步移入新模块，公共文件只做错误边界和 scheduler 结果整合。

这样可以把 queued PP 的 phase 规则放在一个地方，后续删除逻辑时能直接检查所有调用者，避免同一个状态判断在 Engine、Executor 和 Worker 中各维护一份。

### 2. 集中 Worker 侧 queued 运行逻辑

新增 `vllm_omni/diffusion/queued_pp/worker_runtime.py`，承接当前 `DiffusionWorker` 中仅服务于 queued PP 的方法族：

- transport 初始化、reserve/start/poll/release。
- stage queue、active task、awaiting feedback 和 expected receive reservation。
- `progress_pipeline_transfers()`、`_consume_ready_pipeline_message()` 和 receive consumer lease。
- finalization future、device event 和 `WanFinalDecode` executor 的生命周期。

`DiffusionWorker` 只保留一个 queued runtime 成员和少量 RPC 转发函数。普通 diffusion worker 的模型加载、普通 execute、KV cache 和通用错误处理不应被这次清理重新组织。

### 3. 合并 Executor 的重复 RPC 包装

当前 `multiproc_executor.py` 和 `uniproc_executor.py` 都有一组 queued 专用 RPC 包装和结果归一化逻辑。应在新 queued PP 模块中定义统一的 command/result adapter：

- 公共层只定义 `QueuedPipelineExecutor` 所需的窄协议。
- multiprocess 实现负责消息队列、rank 聚合和 future。
- uniprocess 实现负责直接调用同一协议。
- `finalize_pipeline_batch`、`poll_pipeline_finalization`、`release_pipeline_batch` 等方法不再分别复制参数校验和结果展开逻辑。

清理时不能合并普通 `execute_request`、普通 batch forward 或 static PP RPC；这些路径仍是公共能力。

### 4. 缩减 transport 和 schema 的重复状态

下一步应画出每个字段的唯一 owner，再删除只用于转发的重复镜像：

- task identity 只由 `PipelineTask` 持有，message 只携带传输所需的 identity。
- receive reservation、consumer lease 和 send ticket 分别表示三种不同生命周期，不能通过一个布尔字段替代。
- readiness、completion 和 pipeline event 应有单一归一化入口。
- `poll_received(limit)` 的兼容处理和 transport capability 检查只保留在 connector adapter 初始化处。

目标是缩短代码并减少跨文件状态同步，同时保留 stale message 校验、credit 上限、device event 和取消清理。

### 5. 清理测试代码

建议将 queued PP 测试集中到 `tests/diffusion/queued_pp/`，共享一个 CPU fake Worker、fake connector 和 task factory。重点删除以下重复：

- multiprocess/uniprocess 中重复验证同一 RPC 参数的测试。
- connector、Worker、Executor 各自重复构造相同 task identity 的测试。
- 已经被最终 `PipelineStageEngine` 取代的旧 autonomous loop、临时 handoff 和历史 prototype 测试。
- 只验证实现细节、不验证公开 contract 的大量 queue snapshot 测试。

保留四类测试：配置 contract、transport identity/credit、阶段生命周期和数值/失败清理。测试应优先验证状态边界和 observable result，减少对私有字段和轮询次数的断言。

## 清理时的验收门槛

每一次抽取或删除都要同时满足：

1. Wan2.2 PP2 与单卡结果的最大绝对误差保持在 `1e-8` 容差内；相同 seed 的 decoded tensor hash 或逐元素误差必须记录。
2. queued PP2 的 4 请求、1024×1024、80 帧、8/30 steps 回归成功，不能出现额外 finalization failure、stale transfer 或 request hang。
3. 32 请求、8 并发的同构和异构 benchmark 以重复运行的中位数比较；清理不能造成超过测试噪声的吞吐下降，当前 edge2 配置不能退回 edge1 的 credit 行为。
4. 静态 PP2、PP1 和非 queued diffusion 路径的 focused tests 继续通过。
5. 先运行 Ruff、compile 和 focused pytest，再做远端 GPU correctness、trace 和 benchmark；pytest 环境错误必须与代码失败分开记录。

## 明确不纳入本轮清理

- 不引入 VPP 或 virtual chunks。
- 不把 queued PP 调度改造成新的 inter-request scheduler。
- 不修改普通静态 PP 的通信协议。
- 不把 TextEncoder 或 VAE 的通用模型代码搬进 queued PP 模块。
- 不为了减少行数删除 numerical validation、transport identity、device event、取消和 fatal recovery 检查。

推荐的下一次提交处理 Worker transport/runtime 抽取：集中 reserve/start/poll/release 与 receive consumer lease 的窄适配层；保持当前 StageEngine owner 顺序、未启动 send gate、device event 和 fatal recovery contract。
