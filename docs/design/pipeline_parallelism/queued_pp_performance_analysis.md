# Wan2.2 Queued PP 实现与性能分析

更新时间：2026-09-24

本文记录 Wan2.2-TI2V-5B 的 pipeline parallelism 实现进展、验证结果、trace 产物、当前性能瓶颈和后续改动方向。本文中的性能结论基于 PP=2、两张 GPU、无 CFG 的 Wan2.2 长请求 workload。

## 1. 当前结论

queued 模式已经能够在真实 GPU 上完成多请求、多 denoise step 的 PP 执行，并且相较 main 分支的 static step PP 有吞吐收益：平均延迟下降约 16%，吞吐提升约 12%。

但当前实现还不是独立 Worker 持续推进的异步 1F1B pipeline，而是由 Engine 主循环逐轮调用 Worker progress 的排队式交错执行。剩余的主要空泡来自：

1. Engine 在 queued progress 路径中同步执行最终 VAE decode 和 retirement。
2. Worker 的控制面使用默认 NCCL process group 进行大量 `all_gather_object`，rank 之间形成 GPU collective 等待。
3. queued 把 scheduler wave 拆成单请求任务，且 `edge_buffer_slots=1`、`max_inflight_batches=2` 限制了 pipeline 深度和 transfer 并行度。

因此，当前收益主要体现为吞吐改善，而不是每个请求的端到端延迟都下降。

## 2. 已完成的实现

### M1：Wan step execution 与 PP 基础能力

- 从 Wan2.2 的完整 `forward()` 中抽取 request-local 的 preparation、denoise、scheduler update 和 decode 路径。
- 共享 full-forward 与 step execution 的 prompt/embedding normalization、geometry、scheduler 和 latent preparation 逻辑。
- 保持 solver、flow shift、temporal/spatial rounding 和 Wan-specific guidance 行为。
- 添加 PP stage-local execution 能力与真实 PP topology 校验。
- 保持 last-stage 数值更新的 ownership；first stage 持有下一步 latent mirror。
- 修复 PP 非 output rank 的 decode 行为，避免 `None` output 被继续当作 tensor 使用。
- 修复 `AsyncLatents` 在输出和 SHM/pickle 边界上的 materialization。
- 修复 embedding precedence、CFG 行为、VACE capability opt-out、rank-local preparation failure agreement 等问题。

### M2：queued pipeline v1 生命周期

- 添加 queued mode 配置和约束，包括 `max_inflight_batches`、`edge_buffer_slots` 和 stage buffer budget。
- 添加 `PipelineStageState`、`PipelineTask`、`PipelineBatchContext` 和 stage lifecycle。
- 添加 `PipelineStageConnector`、transfer offer/grant、发送 ticket、接收 reservation、consumer lease 和 transport drain。
- 添加 PP=2 distributed P2P activation/feedback transport，保证 P2P 只能在 validated grant 后启动。
- 打通 Engine -> Executor -> Worker -> ModelRunner -> Wan stage execution 路径。
- 添加 queued batch 的 prepare、submit、authorize、progress、cancel、finalize、release、cleanup 和 drain。
- 增加 all-rank topology validation、partial failure agreement、executor fail-closed 行为和 cancellation/retirement retry。
- 增加 rank-local event aggregation、retirement/cancellation acknowledgement validation，以及 stale/duplicate identity 防护。

### M3：多请求和容量扩展

- 允许 `max_num_seqs > 1` 的 queued admission 配置。
- 将 scheduler wave 拆为 one-request task descriptor，维护独立的 queued batch ownership。
- 支持多个 retained batch、capacity deferral、stage buffer byte reservation 和 per-stage memory report。
- 修复 admission capacity exhaustion、cached request deferral、retained batch progress、shared event snapshot 和 queued failure cleanup。
- 修复 finalizing/abort/error/retirement 的持久状态和重试路径。
- 增加 scheduler commit atomicity、duplicate request ID、finalizing request exclusion 和 exact retirement topology 检查。

## 3. 当前 PP 架构与组件边界

### 3.1 总体调用链

当前 queued PP 的实际调用链是 Engine 驱动的同步控制循环：

```text
client request
    -> DiffusionEngine._busy_loop()
    -> Scheduler.schedule()
    -> split into one-request queued descriptors
    -> Engine admission / prepare / submit / authorize
    -> Executor cross-rank control RPC
    -> Worker rank-local stage state
    -> ModelRunner local stage execution
    -> Wan pipeline stage computation
    -> PipelineStageConnector offer / grant / P2P transfer
    -> Worker event and completion
    -> Executor normalization
    -> Engine scheduler commit / final decode / retirement
```

PP 的 logical stage 和 physical rank 是两个不同概念。当前 PP=2 通常是 logical stage 0/1 映射到两个 physical ranks；`PipelineStageSpec` 描述 logical stage，transport offer/grant 使用 physical endpoint，Engine 负责验证二者映射一致。

### 3.2 Engine：全局生命周期和调度驱动者

主要入口：`vllm_omni/diffusion/diffusion_engine.py` 的 `_busy_loop()`、`_run_queued_pipeline_iteration()` 和 `_advance_queued_pipeline_batch()`。

Engine 的职责边界：

- 接收客户端请求并把请求交给 Scheduler admission。
- 从 Scheduler 获取 scheduler output，并将一个 wave 拆成 queued task descriptor。
- 为每个 in-flight batch 分配 `batch_id`、request generation、step index、logical stage map 和 memory reservation。
- 按照 prepare -> submit -> authorize -> progress -> commit -> finalizing -> release 的顺序驱动 Executor。
- 保存 `_queued_pipeline_batches`，处理 capacity deferral、failure、abort、decode、retirement 和最终客户端输出。
- 将一次 shared progress snapshot 按 batch identity 分发给 retained batches，避免同一轮重复 polling。

Engine 不负责：

- 直接调用模型 stage 或操作 device tensor payload；
- 直接调用 `isend/irecv`；
- 在 rank 本地重新做 request admission；
- 代替 Scheduler 修改 request step；
- 把 Worker 的 rank-local context 当作 Engine-owned tensor state。

当前 Engine 仍然在一个 background busy loop 中同步完成一轮 progress、事件处理、final decode 和 retirement，这正是当前主要 bubble 来源之一。它还不是每个 PP stage 独立持续运行的 stage engine。

### 3.3 Scheduler：请求进度和容量的唯一权威

主要实现：`vllm_omni/diffusion/sched/step_scheduler.py`。

Scheduler 的职责边界：

- 维护 request status、waiting/running/finalizing 集合和 request capacity。
- `schedule()` 决定本轮 admission、cached request 和 finished request。
- queued step 完成后，`commit_pipeline_step()` 验证 request identity 集合、step index 和 progress ownership，然后一次性推进 request step。
- 当 step 达到总步数时，只进入 `pipeline_finalizing`；decode 成功、Worker retirement 和 persistent state cleanup 完成后，才允许 request complete。
- 处理 capacity deferral、preempt/defer 和错误完成状态。

Scheduler 不负责：

- PP stage FIFO 或 transfer ordering；
- stage-local tensor 计算；
- P2P grant/credit/lease；
- VAE decode；
- 直接接触 Worker 的 ModelRunner context。

因此，Scheduler 的 step commit 是 Engine 和 Worker 之间的边界：Worker 报告一次完整的 step completion，Engine 再调用 Scheduler commit；Worker 不能自行修改 Scheduler progress，Scheduler 也不能假设 Worker 已经完成 decode 或 transport retirement。

### 3.4 Executor：跨 Worker 的控制面适配器

主要实现：`executor/abstract.py`、`executor/multiproc_executor.py` 和 `executor/uniproc_executor.py`。

Executor 的职责边界：

- 向一个或多个 Worker 发出 prepare/submit/authorize/progress/cancel/finalize/release RPC。
- 收集并规范化各 rank 的返回值，验证 PP endpoint coverage、physical topology 和 acknowledgement identity。
- 维护 transfer coordinator，处理 offer、readiness、grant、start 和 completion。
- 在 partial submission、rank failure、grant timeout、malformed report 或 transport progress failure 时 fail closed。
- 对 multiprocess executor 和 uniprocess executor 提供相同的 queued control contract。

Executor 不负责：

- Scheduler admission 和 request step mutation；
- 保存 request-local latent、solver history 或 ModelRunner state；
- 决定某个 stage 的模型执行顺序；
- 把 rank 0 的结果假设为所有 PP rank 的结果。

Executor 的控制返回必须是 metadata-only 的 descriptor/event/progress。请求 tensor、solver state 和 latent mirror 必须在 Worker/ModelRunner 本地解析和保存，不能通过 queued control RPC 传递 prepared state 对象。

### 3.5 Worker：rank-local stage owner

主要实现：`vllm_omni/diffusion/worker/diffusion_worker.py`。

每个 Worker 只拥有自己 physical rank 对应的 local logical stage，并维护：

- `pipeline_stages`：本地 stage 的 pending/active/awaiting-feedback/terminal 状态；
- `pipeline_send_tickets`：本地发送 payload、credit 和 transfer completion；
- `pipeline_receive_reservations`：接收 credit 从 readiness 到 consumer release 的 ownership；
- `pipeline_pending_received`：已经收到但尚未满足 task identity/FIFO 条件的消息；
- `_pipeline_events`：metadata-only lifecycle events；
- ModelRunner 的本地 `PipelineBatchContext` 引用。

Worker 的职责边界：

- 根据 Engine 下发的 metadata descriptor，从本地 ModelRunner state cache 解析 request state。
- 将 descriptor 放入本地 stage FIFO，并在 authorize 后最多推进一个 local stage task。
- stage 0 执行局部 forward 后产生 activation offer；stage 1 执行局部 forward 和 numerical update 后产生 feedback offer。
- 收到并验证 identity 匹配的 activation/feedback 后，推进下一本地 stage transition。
- 只生成和消费 metadata events；传输 payload 由 Connector/Transport 管理。
- 在取消、失败和 drain 时保留 ownership，直到 ticket/lease/backend operation 安全结束。

Worker 不负责：

- 选择新请求或改变 Scheduler batch membership；
- 跨 rank 重新决定 logical stage mapping；
- 直接完成其他 rank 的 context；
- 在没有 grant 的情况下启动 P2P；
- 因为本地 task 暂时不可运行而丢弃 receive lease。

### 3.6 ModelRunner：request state 和本地计算边界

主要实现：`vllm_omni/diffusion/worker/diffusion_model_runner.py`。

ModelRunner 保存 request-local 的 `StepRequestState`、`InputBatch`、solver/timestep history 和 `PipelineBatchContext`。它提供以下四类操作：

1. `prepare_pipeline_batch()`：从本地 prepared state 创建 context，验证 request/step ownership，并建立 `(request_id, step_index)` owner。
2. `execute_pipeline_stage()`：在 `set_forward_context()` 和 inference scope 中执行一个纯 local partition，不启动 PP transport，也不执行 numerical solver update。
3. `complete_pipeline_step()` / `adopt_pipeline_feedback()`：由最后 stage 负责一次 authoritative scheduler update，由第一 stage 负责接收反馈并更新 latent mirror。
4. `finalize_pipeline_batch()`：由 output owner 调用 Wan `post_decode()`，生成最终输出；之后才能进入 release/cleanup。

ModelRunner 不负责 Scheduler commit、P2P offer/grant、跨 rank event aggregation 或客户端输出流。它只保证 context 状态、request step identity、inference semantics 和本地 tensor ownership 正确。

### 3.7 Wan pipeline：模型特定的 preparation、stage forward、solver 和 decode

主要实现：`vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py`。

Wan pipeline 的边界是模型计算，而不是运行时调度：

- `prepare_encode()` / shared preparation helpers 创建 prompt embeddings、negative conditioning、scheduler、timesteps 和 initial latents。
- `forward_pipeline_stage()` 只执行当前 PP partition 的 local transformer computation。
- `step_scheduler_pipeline_stage()` 只在 last stage 执行一次 numerical update。
- `post_decode()` 只处理 request-local latents 到最终 video/output 的转换。
- Wan transformer 根据真实 PP topology 选择 first/last partition；不能由外部传入一个与实际 rank 不一致的 spec。

Wan pipeline 不应：

- 自己调用 queued transport；
- 自己推进 Engine/Scheduler；
- 把 PP feedback 再次当作 solver update；
- 依赖另一个 request 的 mutable pipeline-global state。

### 3.8 Connector 和 Transport：受 grant 约束的传输 ownership

主要实现：`vllm_omni/diffusion/distributed/pipeline_stage_connector.py`。

`PipelineStageConnector` 位于 Worker 和具体 backend 之间，负责：

- directed edge validation、per-edge FIFO 和 transfer identity；
- send credit、receive reservation、send/receive ticket；
- offer -> readiness -> grant -> start -> completion -> consumer release 生命周期；
- duplicate/stale identity、replay tombstone、close/drain 和 backend ownership；
- 将 backend payload 隔离为 metadata message + tensor dictionary。

`DistributedP2PTransport` 只在 validated grant 后调用 `isend_tensor_dict()` / `irecv_tensor_dict()`。Connector/Transport 不知道 Scheduler、模型结构或 request admission，也不能绕过 Worker 的 task identity 检查。

### 3.9 一次 queued denoise step 的 ownership 转移

```text
Scheduler.schedule()
  -> Engine reserves batch/request ownership
  -> Executor.prepare_pipeline_requests()
  -> Worker resolves local state from ModelRunner cache
  -> Executor.submit_pipeline_batch()
  -> Worker creates stage-local context and FIFO task
  -> Executor.authorize_pipeline_batch()
  -> Worker stage 0 executes local forward
  -> Connector reserves activation send and waits for grant
  -> stage 1 receives activation and executes local forward
  -> last stage completes numerical solver update
  -> Connector sends feedback latent to stage 0
  -> stage 0 adopts feedback and emits STEP_COMPLETED
  -> Engine calls Scheduler.commit_pipeline_step()
  -> final step enters FINALIZING, then output owner decodes
  -> transfer leases retire, contexts release, scheduler request completes
```

这个顺序体现了三个独立 contract：

1. Scheduler progress 只能由 Engine 在完整 step completion 后提交。
2. Tensor transfer 只能由 Connector 在 grant 后启动，且直到 consumer/backend completion 才释放 ownership。
3. ModelRunner context 只能在 terminal stage status、transfer retirement、decode/cleanup 条件都满足后释放。

### 3.10 当前架构的明确限制

当前实现不是每个 PP stage 独立拥有持续事件循环的真正异步 1F1B runtime。Worker 虽然拥有 rank-local stage state 和 bounded transport ownership，但 progress 仍由 Engine 主循环按 round 触发；Executor 的跨 rank control 也仍然是同步 RPC/collective 边界。

因此，当前架构可以实现有限深度的 inter-request stage overlap，但仍保留：

- Engine round barrier；
- final decode/retirement barrier；
- control metadata collective barrier；
- bounded FIFO/credit barrier。

后续异步化必须保持上述 ownership 边界，不应通过复制 Scheduler、让 rank-local Worker 自行 admission，或让 Wan pipeline 直接持有 transport 来规避这些边界。

### 3.11 progress RPC 的 rank-wide barrier

需要特别区分“Worker 已经完成本次 local forward”和“Engine 已经可以发起下一轮 progress RPC”。当前二者不是同一时刻。

当 stage 0 和 stage 1 各自有非空队列，并且一次 progress 中分别处理请求 A/B 时，实际时序是：

```text
Engine 发起 progress_pipeline()
  -> rank 0 推进 stage 0 的 A
  -> rank 1 推进 stage 1 的 B
  -> Worker rank agreement / RPC response
  -> Executor 汇总两个 rank 的结果
  -> Engine 消费事件、处理 transfer、提交 scheduler step
  -> Engine 发起下一次 progress_pipeline()
```

如果 stage 0 的 A 先完成，而 stage 1 的 B 还需要一段时间：

```text
rank 0: A local forward 完成，发送 activation，等待本轮 RPC 结束
rank 1: B 继续执行 local forward / numerical update
rank 1: B 完成后返回
Executor: 等待并汇总两个 rank
Engine: 此时才能发起下一轮 RPC
rank 0: 才能开始队列中的下一个任务 C
```

因此这个场景中实际空闲的是已经完成 A 的 stage 0；stage 1 在继续执行 B。stage 0 虽然已经释放了本地 active slot，仍不能在同一轮 RPC 内立即执行 C，因为当前 `progress_pipeline_transfers()` 每次最多推进一个本地 task，且 `_run_and_gather_rank_values()` 会在 Worker 返回前等待所有 rank 的结果。

这也是当前实现与真正异步 1F1B 的关键区别：Worker 内部有 stage-local FIFO 和 transport ownership，但 progress clock 仍由 Engine 的同步 round 驱动。

### 3.12 Gloo control group 不能单独消除 round barrier

将 metadata agreement 从默认 NCCL process group 切换到 PP CPU/Gloo group 是必要的控制面修正，但不是完整的流水线解耦方案。

它可以解决：

- 小型 metadata 不再占用 GPU/NCCL collective；
- rank 1 不再把长时间的控制等待表现为 NCCL GPU kernel；
- control collective 的 GPU trace 污染和部分同步开销。

它不能解决：

```text
Worker progress
  -> rank agreement
  -> Executor 等待所有 rank
  -> Engine 才能发起下一轮 progress
```

只要 Executor 仍然把 progress 当成一个等待所有 Worker 返回的 collective RPC，stage 0 就仍然不能在 stage 1 尚未完成时推进 C。

### 3.13 推荐的异步化边界：Worker 内部 StageEngine

不建议把 PP stage 的模型执行和 device ownership 提升到 Engine 进程。Engine 不应持有：

- GPU model partition；
- request-local latent、solver history 和 `PipelineBatchContext`；
- CUDA event、P2P tensor buffer 和 transport lease。

更合适的目标架构是保留 Engine 作为全局 control plane，在每个 Worker 内增加持续推进本地 stage 的 StageEngine：

```text
Engine:
  admission / request identity / scheduler commit
  cancel / finalization / retirement

Worker rank 0 StageEngine:
  stage0(A) -> stage0(C) -> stage0(E) -> ...

Worker rank 1 StageEngine:
  stage1(B) -> stage1(D) -> stage1(F) -> ...

Executor:
  submit descriptors
  asynchronously collect metadata events
```

StageEngine 应继续由 Worker 拥有以下状态：

- pending/active/awaiting-feedback FIFO；
- local ModelRunner contexts；
- send tickets、receive reservations 和 consumer leases；
- CUDA completion events 和 transport completion；
- metadata-only stage progress events。

Engine 只异步消费这些事件，并在完整 step completion 后调用 Scheduler commit。这样可以让 stage 0 在 stage 1 继续处理 B 时推进 C，同时不复制 Scheduler，也不让 rank-local Worker 自行 admission。

### 3.14 当前框架内的渐进改法

可以分两个阶段降低风险：

1. **先移除 hot path 的同步 rank gather。** `progress_pipeline_transfers_all_ranks()` 不再在 Worker 内部通过 `_run_and_gather_rank_values()` 等待所有 rank；Executor 改为分别收集 rank-local response，或使用 per-rank response queue。
2. **再增加 Worker 内部持续 progress loop。** Worker 在输入 descriptor、transport event 和 device completion 满足时自主推进下一个 FIFO task，Engine 只处理 submit、commit、cancel、finalization 和 retirement。

无论采用哪种实现，都必须保持以下约束：

- 同一 request 的 step dependency 不能越过；
- 同一 directed edge 仍然遵循 FIFO；
- P2P 必须先经过 validated grant 才能启动；
- send/receive ownership 不能在 consumer/backend completion 前释放；
- Scheduler commit 只能由 Engine 在完整 step completion 后执行；
- final decode 和 retirement 仍然由显式状态机控制。

## 4. 关键提交和工具

关键实现提交包括：

| 提交 | 内容 |
| --- | --- |
| `f5e199021` | Wan2.2 step execution 基础实现 |
| `c1d0c9b20` | Wan preparation 和 CFG capability 修复 |
| `e3ac647b0` | queued pipeline stage connector |
| `f1efa605` | ModelRunner pipeline contexts |
| `316691186` | Worker pipeline lifecycle |
| `0723adb46` | connector lifecycle hardening |
| `bb4347644` | queued pipeline control routing |
| `3af3d7ec3` | granted queued P2P transport |
| `51946bd37` | transfer grant wiring |
| `89057fc20` | Worker transfer progression |
| `5553e6626` | Executor/Worker progress coordination |
| `c61679472` | Engine queued lifecycle |
| `dd7501ea9` | cached queued pipeline steps |
| `e6ca8993f` | memory admission and PP budget reports |
| `dafb9fe56` | queued PP multi-batch stage progress |
| `81d60a48a` | admission deferral 时保持已有 queued ownership |
| `c28f8684e` | Wan2.2 queued PP benchmark runner |
| `42bb7498` | queued PP trace visualization |
| `c9736a4c2` | profiler timeline origin normalization |

当前本地 HEAD：`c9736a4c2`。

新增 benchmark 工具：

- `tools/benchmark/run_wan22_queued_pp.py`
- `tools/benchmark/visualize_wan22_queued_pp.py`

静态验证记录：

- focused diffusion/queued suites 在迭代过程中持续通过；最新记录为 299 passed，个别 benchmark 修复快照为 298 passed。
- Ruff check、Ruff format check 和 `git diff --check` 通过。
- 真实 GPU/NCCL workload 通过 `canhazgpu` 申请资源运行；所有 4 个请求都成功完成，退出时 GPU 已释放。

## 5. 验证环境和 workload

测试遵循远程 GPU 流程：通过 `canhazgpu` 申请 GPU，在 `lab-dell03` 上进入 `~/vllm-omni-disag-dit`，同步目标分支，使用 Python 3.12 的 `.venv`。远程依赖由 uv 管理，使用 `uv pip`，不使用普通 `pip`。

对比 workload：

```text
model: Wan2.2-TI2V-5B-Diffusers
pipeline_parallel_size: 2
step_execution: true
mode: queued / static
max_num_seqs: 2
max_inflight_batches: 2
edge_buffer_slots: 1
request_count: 4
resolution: 512 x 512
frames: 16
denoise steps: 8
guidance_scale: 1.0
seeds: 42, 43, 44, 45
arrival_interval_ms: 0
enforce_eager: true
```

queued 与 main/static 使用相同模型、请求参数和 PP=2 拓扑。Profiler 运行只用于诊断，非 profiler 运行用于最终延迟和吞吐结论。

## 6. 性能结果

### 非 profiler 结果

三轮 queued 和三轮 static 运行，每轮 4 个请求，共 12 个请求样本：

| 指标 | static | queued | 变化 |
| --- | ---: | ---: | ---: |
| 平均端到端延迟 | 4614 ms | 3874 ms | -16.0% |
| P50 延迟 | 3276 ms | 3144 ms | -4.0% |
| P95 延迟 | 6274 ms | 5463 ms | -12.9% |
| 平均 wave throughput | 0.657 req/s | 0.736 req/s | +11.9% |
| 延迟标准差 | 1530 ms | 1283 ms | queued 较低 |

请求级别表现并不均匀。一个代表性 workload 中：

| request | static | queued |
| --- | ---: | ---: |
| request 0 | 约 3132 ms | 约 5437 ms |
| request 1 | 约 3175 ms | 约 4632 ms |
| request 2 | 约 6062 ms | 约 2313 ms |
| request 3 | 约 6088 ms | 约 3114 ms |

这说明 queued 主要改善后续请求的 overlap 和总吞吐，但首批请求承担了 pipeline fill、final decode 和 control synchronization 的代价。

### Profiler trace 统计

Profiler 运行的 elapsed/latency 受到 profiler 影响，不用于最终性能结论。其主要用途是定位空泡：

| 指标 | static rank 0 | static rank 1 | queued rank 0 | queued rank 1 |
| --- | ---: | ---: | ---: | ---: |
| GPU idle union | 32.3% | 58.9% | 24.3% | 25.4% |
| GPU span | 6.069 s | 5.060 s | 5.819 s | 5.838 s |
| gaps >= 5 ms | 26 | 27 | 12 | 12 |

queued 明显降低了 static 中 rank 1 的空闲比例，但该统计会把长时间等待的 NCCL collective kernel 计为 GPU busy，因此不能单独代表有效计算利用率。

## 7. Trace 和结果文件位置

所有下载到本地的完整结果位于：

```text
artifacts/wan22-long-comparison-42bb7498/
```

重要文件：

- `queued-profile-manifest.json`：queued profiler 运行 manifest。
- `main-profile-manifest.json`：main/static profiler 运行 manifest。
- `queued-latency-repeat{1,2,3}-manifest.json`：queued 非 profiler 结果。
- `main-latency-repeat{1,2,3}-manifest.json`：static 非 profiler 结果。
- `queued-profile-torch_profiler/.../trace_rank{0,1}.json`：queued 两个 rank 的 PyTorch trace。
- `main-profile-torch_profiler/.../trace_rank{0,1}.json`：static 两个 rank 的 PyTorch trace。
- `queued-profile-worker_events/worker_rank{0,1}.jsonl`：queued Worker 生命周期、stage progress 和 transfer control 事件。
- `wan22_pp_timeline.svg`：rank/stage timeline 可视化。
- `wan22_latency_comparison.svg`：三轮非 profiler 延迟对比图。
- `wan22_comparison_summary.json`：可机器读取的汇总结果。

可视化文件：

- [PP timeline](../../../artifacts/wan22-long-comparison-42bb7498/wan22_pp_timeline.svg)
- [Latency comparison](../../../artifacts/wan22-long-comparison-42bb7498/wan22_latency_comparison.svg)
- [Summary JSON](../../../artifacts/wan22-long-comparison-42bb7498/wan22_comparison_summary.json)

## 7.1 可复现实验命令

先在已通过 `canhazgpu` 分配的远程 GPU 环境中完成仓库同步和虚拟环境激活：

```bash
cd ~/vllm-omni-disag-dit
git pull origin feature/pipeline_parallelism
source .venv/bin/activate
```

依赖变更使用 uv 管理，例如：

```bash
uv pip install <package>
```

queued profile：

```bash
python tools/benchmark/run_wan22_queued_pp.py \
  --model ~/models/Wan2.2-TI2V-5B-Diffusers \
  --mode queued \
  --output-dir /tmp/wan22-queued-profile \
  --max-num-seqs 2 \
  --max-inflight-batches 2 \
  --edge-buffer-slots 1 \
  --request-count 4 \
  --height 512 --width 512 --num-frames 16 \
  --num-inference-steps 8 --seed 42 --profile
```

static profile 只需将 `--mode queued` 改为 `--mode static`，并使用不同的 output directory。非 profiler 延迟重复实验去掉 `--profile`，分别运行三轮。

trace 分析：

```bash
.venv/bin/python .agents/skills/diffusion-perf-opt/scripts/trace_analyzer.py \
  /path/to/trace_rank0.json
```

生成 timeline 和延迟图：

```bash
python tools/benchmark/visualize_wan22_queued_pp.py \
  --queued-profile /path/to/queued-profile \
  --main-profile /path/to/main-profile \
  --queued-run /path/to/queued-latency-repeat1/manifest.json \
  --queued-run /path/to/queued-latency-repeat2/manifest.json \
  --queued-run /path/to/queued-latency-repeat3/manifest.json \
  --main-run /path/to/main-latency-repeat1/manifest.json \
  --main-run /path/to/main-latency-repeat2/manifest.json \
  --main-run /path/to/main-latency-repeat3/manifest.json \
  --output-dir artifacts/wan22-long-comparison-42bb7498
```

## 8. 当前空泡的完整原因分析

### 8.1 最主要的空泡：final decode 串行化

queued batch 在最后一个 denoise step 完成后，会进入：

```text
STEP_COMPLETED
  -> Engine._advance_queued_pipeline_batch()
  -> _finalize_queued_pipeline_batch()
  -> Executor.finalize_pipeline_batch()
  -> Worker.finalize_pipeline_batch()
  -> ModelRunner.finalize_pipeline_batch()
  -> WanPipeline.post_decode()
  -> VAE decode
```

相关代码：

- `vllm_omni/diffusion/diffusion_engine.py:_advance_queued_pipeline_batch()`
- `vllm_omni/diffusion/worker/diffusion_model_runner.py:1280`
- `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py:1226`

`post_decode()` 在 VAE decode 前还会调用 `empty_cache()`。同时，Worker 通过 `_run_and_agree_rank_status()` 让非 output rank 参与 rank agreement。于是：

1. rank 0 执行 `empty_cache()` 和 VAE decode。
2. rank 1 已经进入默认 NCCL `all_gather_object`，等待 rank 0。
3. Engine 在 finalization/retirement 完成前不会进入下一次 pipeline progress。
4. stage 0 和 stage 1 都暂时失去可推进的任务。

queued trace 中 rank 1 出现 4 次约 680–692 ms 的 NCCL all-gather 等待，正好对应 4 个请求的 finalization。该 collective 的消息只有 1 个 Long 或几十个 Byte，不可能是有效的大规模数据传输，主要是 rank skew 导致的同步等待。

### 8.2 第二个瓶颈：控制面使用默认 NCCL metadata collective

`_all_gather_rank_values()` 使用未指定 group 的 `dist.all_gather_object()`：

```text
_run_and_gather_rank_values()
  -> _all_gather_rank_values()
  -> dist.all_gather_object()  # default process group
```

该 helper 被以下路径频繁调用：

- `progress_pipeline_transfers_all_ranks()`
- readiness acceptance
- grant/start
- event polling
- cancellation/release/cleanup
- final decode status agreement

queued rank 1 trace 中约有 1318 次 all-gather，累计 NCCL kernel 时间约 2.847 s，最大单次约 691.7 ms。真正的 PP activation/feedback P2P 只有几十次，单次约 0.1 ms 以内，因此目前主要问题不是 activation tensor 的 P2P 带宽或 NCCL send/recv。

当 rank 1 先进入 metadata collective，而 rank 0 仍在 final decode 或上一轮 Engine control path 中时，rank 1 的 NCCL kernel 会持续等待。它在 profiler 中表现为 GPU busy，但实际上没有有效模型计算。

### 8.3 Engine progress 是单线程、单轮驱动

当前 Engine 主循环大致是：

```text
progress_pipeline()
poll_pipeline_events()
处理事件
commit scheduler step
可能执行 final decode
可能执行 retirement
下一轮 progress_pipeline()
```

因此存在一个全局串行边界：Engine 必须完成当前轮的控制、事件处理和可能的 decode，才能调用下一轮 progress。Worker 没有独立的持续 stage clock，也不能在 Engine 等待 final decode 时继续推进其他 stage-local work。

### 8.4 单请求 descriptor 没有利用 request batching

`_split_queued_scheduler_output()` 会把一个 scheduler output 拆成单请求 descriptor。于是 queued 模式中 `max_num_seqs=2` 主要影响 admission capacity，不代表一次 denoise 调用一定处理两个请求。

相比之下，static path 可以在同一次模型调用中处理两个请求，摊销：

- input preparation；
- scheduler/solver bookkeeping；
- RPC 和 event handling；
- 模型 launch 和 kernel setup。

queued 的 inter-request overlap 抵消了部分 batch amortization 收益，因此吞吐有所提升，但单请求延迟没有成比例下降。

### 8.5 pipeline 深度和 transfer slot 太浅

当前 workload 使用：

```text
max_inflight_batches = 2
edge_buffer_slots = 1
```

每条 activation/feedback edge 同时只允许一个 transfer ticket。stage 计算约为 rank 0 35 ms、rank 1 44 ms，但稳定的 stage marker 启动间隔约为 68–70 ms，说明计算之间还有明显的控制、credit 和 FIFO 等待。

增加 slot 或 in-flight depth 可能改善填充，但不能单独消除 final decode barrier，也必须受真实 stage buffer memory 限制。

### 8.6 startup/flush 和请求公平性

4-request workload 的请求数量不大，pipeline fill/flush 占比仍然明显。Worker event 序列还显示不同请求在不同阶段推进速度不一致，部分请求会连续推进多个 step，而其他请求等待 FIFO、feedback 或 capacity。

因此需要分别报告：

- 首请求延迟；
- steady-state request latency；
- makespan；
- wave throughput；
- stage useful compute utilization。

不能只用平均延迟判断 queued 是否成功。

## 9. 当前结果的限制

### 9.1 输出质量尚未完成严格 A/B

static 和 queued manifest 使用不同 commit：

- queued：`81d60a48a`
- static：`333f14e9f`

相同 seed 的 decoded SHA 也不同。由于运行进程、调度顺序、CUDA kernel 执行顺序和数值非确定性可能不同，hash 不一致本身不能证明输出质量错误；但当前结果也不能被称为严格的数值等价 A/B。

后续应固定相同 source revision 和运行配置，并比较：

- latent trajectory 的 max absolute error / mean absolute error；
- decoded video 的 MAE、PSNR、SSIM；
- 必要时比较每个 denoise step 的 latent checkpoint。

### 9.2 Profiler latency 不能用于性能结论

Profiler 会改变 launch、同步和 CPU scheduling 行为。最终性能数字应继续使用非 profiler manifest；trace 只用于解释空泡和定位调用链。

### 9.3 `empty_cache()` 的精确成本仍需单独测量

目前 trace 已经证明 finalization 周围存在大等待，但还没有把等待精确拆成 `empty_cache()`、VAE kernel、CPU output conversion 和 status collective。需要独立 marker 和 A/B 运行后再决定是否修改 allocator policy。

## 10. 预期改动方向

### P0：先补齐可归因性

在 finalization 路径增加独立 trace marker：

```text
queued.finalize.enter
queued.post_decode.enter
queued.empty_cache
queued.vae_decode
queued.output_prepare
queued.finalize.status_collective
queued.finalize.exit
```

同时给 control collective 增加 operation name、logical stage、batch ID 和 step index，避免只看到匿名的 NCCL all-gather。

需要做的 A/B：

1. `output_type=latent`，隔离 VAE decode。
2. 保留 VAE decode、去掉或延迟 `empty_cache()`。
3. 保留 `empty_cache()`，只测 VAE decode。
4. 把 control agreement 切换到 PP CPU/Gloo group。

### P1：移除默认 NCCL metadata agreement

候选方案：

1. Worker 内的 PP-local failure/event agreement 使用 `get_pp_group().cpu_group`。
2. 或者 Worker 只返回 rank-local RPC result，由 Executor 收集和校验 rank coverage。
3. 保留 device NCCL 只用于真正的 tensor communication，不用于小型 metadata/control object。

需要特别检查 DP/TP/PP 混合拓扑，不能简单把所有 default group collective 都替换成 PP group。

预期收益：消除 rank 1 大量等待型 NCCL kernel，降低每轮 progress 的同步成本。

### P1：解除 progress RPC 的 rank-wide barrier

仅把 metadata collective 换成 Gloo 不能解决 Engine 必须等待所有 Worker 返回的问题。当前一次 progress 的改动方案应分为两个阶段：

#### 阶段 A：rank-local progress 和异步结果收集

将当前：

```text
Engine -> 一个 collective progress RPC
      -> 所有 Worker 执行 local progress
      -> Worker 内 all-gather
      -> Executor 等所有 rank 返回
```

改为：

```text
Engine/Executor -> 各 rank 的 progress command
Worker rank 0   -> 独立推进自己的 stage 并上报 event
Worker rank 1   -> 独立推进自己的 stage 并上报 event
Executor        -> 异步收集 rank-local event/progress
Engine          -> 只在收到完整 STEP_COMPLETED 时 commit scheduler
```

具体改动：

1. `progress_pipeline_transfers()` 保留为 rank-local 操作，不在热路径中调用 `_run_and_gather_rank_values()`。
2. `MultiprocDiffusionExecutor` 增加 per-rank response/event queue 或等价的异步 response collector，不要求 rank 0 和 rank 1 同时完成当前 local forward。
3. Executor 仍然验证 event 的 physical rank、logical stage、batch identity 和 completion coverage；但 coverage validation 应作用于需要成对确认的 transfer/step completion，而不是阻塞每一次 local progress。
4. failure agreement 从每轮 progress 中移出，只在 submit、grant start、cancel、release 和明确的 terminal boundary 执行；progress 失败通过 rank-local error event 进入 Engine 的 fail-closed cleanup。
5. Engine 不再把“没有收到某个 rank 的本轮 response”当作该 rank 已经空闲；未完成的 Worker work 继续留在 Worker-owned state 中，由后续 event 或 completion poll 驱动。

阶段 A 的验收标准是：stage 0 完成 A、stage 1 仍在执行 B 时，stage 0 能够开始 C；Engine 不需要等待 B 才能发送下一条 stage-0 command。

#### 阶段 B：Worker 内部 StageEngine

阶段 A 仍然可能产生较多 Engine/Executor command 往返。稳定版本应在 Worker 内增加持续运行的 StageEngine：

```text
Worker StageEngine loop:
  poll transport completions
  poll CUDA/device completions
  release finished leases
  if stage input/credit/context ready:
      start next FIFO local task
  publish metadata-only progress/event
```

StageEngine 的输入是 Engine 下发的 metadata descriptor、authorize/cancel/retire command 和 transport arrival；输出是 metadata-only event。它不拥有 Scheduler，也不决定新的 request admission。

需要新增或扩展的 Worker 状态：

- `active_device_work` 或等价的 CUDA completion record，区分“local forward 已提交”和“local forward 已完成”；
- stage-local wakeup/condition，使 activation arrival、send completion、receive completion 和新 descriptor 都能唤醒本地推进；
- event sequence/identity，保证同一 `(batch_id, step_index, epoch, branch)` 只发布一次 terminal event；
- stage-local backpressure 状态，保证 edge credit、stage buffer budget 和 consumer lease 不被绕过。

Engine 在阶段 B 中只保留以下同步点：

```text
admission/submission
  -> complete STEP_COMPLETED
  -> scheduler commit
  -> finalization/decode completion
  -> retirement acknowledgement
```

它不再为每一个 local forward 调用一次 broadcast progress RPC。

#### 阶段 A/B 的共同不变量

- 同一 request 的 denoise step 必须按序完成，不能因为 stage-local 独立推进而越过 scheduler commit。
- 同一 directed edge 继续遵循 FIFO；不同 edge 或不同 stage 可以并行。
- P2P 仍然只能在 validated grant 后启动。
- tensor payload、solver history、CUDA event 和 transfer lease 继续由 Worker/ModelRunner/Connector 拥有。
- Engine 只消费 metadata event，并在完整 step completion 后修改 Scheduler 状态。
- cancel、failure、final decode 和 retirement 必须保留可重试、fail-closed 和 drain 语义。

预期收益是消除“stage 0 已完成但等待 stage 1 返回”的空泡，而不是简单增加 RPC 并发数。评估时应分别测量 local forward useful time、Engine/Executor wait time、control collective wait time 和 stage idle time。

### P1：将 final decode 从 progress critical path 解耦

建议引入显式的 finalization state：

```text
STEP_COMPLETED
  -> FINALIZING
  -> decode submitted
  -> decode completed
  -> release/cleanup
  -> scheduler completion
```

Engine 不应在一个 `_advance_queued_pipeline_batch()` 调用中同步完成 decode、retirement 和 scheduler completion。可行方向包括：

- final decode asynchronous submission；
- decode future/status polling；
- output-owner 独立 decode queue；
- decode 期间继续推进其他未 finalizing 的 stage-local tasks。

必须保持：

- decode owner 和 output rank 语义不变；
- release 不能早于 transfer/consumer retirement；
- scheduler completion 不能早于 decode 成功和 ownership release；
- failure/abort 仍然可以重试和 fail-closed。

### P1：增加 pipeline 深度

在显存允许时测试：

```text
edge_buffer_slots = 2, 4
max_inflight_batches = 4
```

必须同步观察：

- stage buffer reservation；
- send/receive ticket 数量；
- retained latent 和 intermediate tensor bytes；
- OOM 和 drain correctness；
- stage useful utilization。

这属于填充优化，不能代替 final decode 和 control collective 解耦。

### P2：重新评估任务粒度和 batch amortization

queued 当前是单请求 descriptor。后续可以研究：

- 对兼容请求保留有限 batch；
- stage-local 处理多个兼容 request row；
- 将 request admission、in-flight batch 和 transport ticket 的容量单位进一步分离。

这需要重新验证 request-local solver state、生成器、guidance、取消和错误隔离，不能直接恢复 static 的整批执行。

### P2：VAE 优化

在控制面和异步 finalization 稳定后再评估：

- VAE patch/spatial parallel；
- decode workspace reuse；
- VAE kernel/compile 优化；
- 避免每个请求强制 `empty_cache()`。

## 11. 后续验证矩阵

每一项优化都应使用相同模型、commit、seed、请求顺序和 remote `canhazgpu` 环境，至少包含：

| 实验 | 目的 |
| --- | --- |
| static vs queued，4 requests | 基线趋势 |
| queued，8/16 requests | startup/flush 与 steady state 分离 |
| queued，slots 1/2/4 | transfer backpressure 影响 |
| queued，in-flight 2/4 | pipeline depth 影响 |
| latent output vs decoded output | 分离 VAE 成本 |
| Gloo control vs default NCCL | 验证 metadata collective 瓶颈 |
| sync final decode vs async final decode | 验证最大 bubble 来源 |
| numerical parity | 验证输出质量 |

每轮应同时记录：

- mean/P50/P95 latency；
- first-request latency；
- makespan；
- throughput；
- per-rank useful compute time；
- control collective wait time；
- VAE decode time；
- transfer ticket/lease occupancy；
- memory peak 和 retained bytes。

## 12. 当前状态

当前实现已经完成 queued PP 的真实 GPU 功能验证、多请求性能基线，以及第一轮高频控制面优化。PP=2 下，rank-local progress/readiness RPC 已由 Executor 收齐各 Worker 的本地结果，不再为每次 progress 和 transfer readiness 都在 Worker 内执行 PP `all_gather_object`。这降低了控制等待并改善了同配置 queued 性能，但没有消除 Engine 驱动轮询、剩余 Gloo collective 和 stage 空泡。

因此仍不应声称“独立 Worker 异步 1F1B 已实现”。下一阶段应聚焦：

1. 给无进展的 Engine rounds 做事件驱动等待/退避，避免高频空轮询；
2. 定位 profile 中剩余长 Gloo wait 的具体 RPC/collective，并评估将剩余 metadata 汇总移到 Executor；
3. 继续拆分 stage 1 idle 中的 P2P 等待、stage 0 计算和 VAE decode 贡献；
4. 评估扩大 slot/depth 的吞吐收益和显存代价；
5. 完成 static/queued 数值质量 parity 和多轮稳定性评估。

当前最准确的能力描述是：

> queued 模式已经支持 PP=2 下的多请求 step execution，并通过有限深度的 Engine-driven interleaving 提高吞吐；它仍然受同步控制轮询、剩余控制 collective、VAE/finalization 和有限 pipeline depth 的限制。

## 13. 2026-09-28：PP Rank-Local 控制 RPC

### 改动边界

PP=2、DP=1、TP/SP/CFG=1 时，`MultiprocDiffusionExecutor` 对 progress 和 transfer readiness 使用 rank-local RPC：

- 每个 Worker 独立返回本地 `PipelineTransportProgress`、已缓冲 events 或 readiness report；
- Executor 从各 Worker result queue 收齐响应，验证 worker identity、PP rank 覆盖和状态后再更新 coordinator；
- rank-local RPC 中任一 Worker 报错时，先收齐本轮各 Worker 响应，再 fail-closed，避免剩余结果串到下一轮；
- 不满足上述拓扑时继续走原来的 all-rank Worker 聚合路径。

该边界只覆盖高频 progress/readiness 路径；初始化、请求准备、retirement、cleanup 等低频或需要 rank agreement 的操作仍保留原协议。相关实现和回归位于 `vllm_omni/diffusion/executor/multiproc_executor.py`、`vllm_omni/diffusion/worker/diffusion_worker.py` 及对应 executor/worker 测试。

### Profile 对比

两份 trace 均为 Wan2.2-TI2V-5B、PP=2、4 请求、512x512x16、8 steps、seed 42。Profile 只用于诊断，不作为 latency 基准。

| 指标 | rank-local 前 | rank-local 后 |
| --- | ---: | ---: |
| Rank 0 GPU idle | 37.58% | 30.21% |
| Rank 1 GPU idle | 72.45% | 64.21% |
| Rank 1 `gloo:all_gather` 次数 | 894 | 666 |
| Rank 1 Gloo annotation 累计时长 | 2.029 s | 0.524 s |
| Rank 1 最大 Gloo annotation | 343.5 ms | 273.5 ms |

Profile artifacts：

- rank-local 前：`artifacts/wan22-queued-merged-progress-profile-20260928/`
- rank-local 后：`artifacts/wan22-queued-rank-local-profile-20260928/`

两份 profile manifest 记录的远端 HEAD 也为 `a82f29b1`（pull 前）；运行时源码已同步并与提交 `69799427` 一致。Profile latency 仍只作诊断，提交边界验证见下方 clean-checkout smoke。

主要剩余空泡包括 rank 1 约 741 ms 的 GPU idle gap，以及约 274 ms 的 Gloo annotation。Rank-local 优化减少了高频同步，但该长 Gloo wait 的具体调用来源仍需继续归因；trace 也不证明其中全部是纯 collective 等待。

### 非 Profiler A/B

每次均通过远端 `canhazgpu run --gpus 2`，模型为 `~/models/Wan2.2-TI2V-5B-Diffusers`，配置为 `max_num_seqs=2`、`max_inflight_batches=2`、`edge_buffer_slots=1`、4 并发请求、512x512x16、8 steps、seed 42。无 warmup；这些是同一固定 workload 的单波次测量，因此用三次样本的中位数比较，不把 profiler 运行混入 latency。

| 版本 | 三次 makespan (s) | 三次吞吐 (req/s) | 中位 makespan (s) | 中位吞吐 (req/s) | 中位 P50 延迟 (ms) |
| --- | --- | --- | ---: | ---: | ---: |
| rank-local 前 | 6.736, 6.011, 6.610 | 0.594, 0.665, 0.605 | 6.610 | 0.605 | 3720 |
| rank-local 后 | 5.770, 5.847, 5.955 | 0.693, 0.684, 0.672 | 5.847 | 0.684 | 3406 |

相对紧邻版本的三次中位数，吞吐提高约 13.1%，makespan 降低约 11.6%，P50 降低约 8.5%。三次运行的四个 decoded SHA-256 均逐请求一致：

```text
404d56baef91c6bcb1cfd9ee9dcf646aa444631c0f3aae703672e41806142fcd
7f8f1b131991dd5881bb5dcb38ec86696a899f560584ca02d092ef01ac55f593
acb20bd3b9061f4fcf1069b6c03a44cb60d7533e40568214153bc9d9dd32bcf5
29a3ce974fbf2877275f1c42fc922cb8de9e1d1223eaeee5a0bbeb2d6d477c7a
```

结果目录：

- rank-local 前：`artifacts/wan22-queued-merged-progress-noprof-20260928-r1/`、`-r2/`、`-r3/`
- rank-local 后：`artifacts/wan22-queued-rank-local-noprof-20260928-r1/`、`-r2/`、`-r3/`

上述六份 manifest 的 `git_revision` 仍记录远端 checkout 基线 `a82f29b1`，因为测量发生在 push/pull 前的远端工作树。与运行相关的源码文件当时已逐一与最终提交内容校验一致；输出哈希也全部一致。随后在 clean checkout 的提交 `69799427` 上又做了一次独立 smoke，该样本不计入上表三次统计：makespan 5.826 s、吞吐 0.687 req/s、mean latency 4217 ms、P50 3330 ms、P95 5826 ms，4 个 decoded hash 相同。其 artifact 位于 `artifacts/wan22-queued-committed-smoke-20260928/`。

复现命令模板：

```bash
canhazgpu run --gpus 2 -- \
  bash -lc 'cd ~/vllm-omni-disag-dit && source .venv/bin/activate && \
  python tools/benchmark/run_wan22_queued_pp.py \
    --mode queued --output-dir /tmp/<run-id> \
    --max-num-seqs 2 --max-inflight-batches 2 --edge-buffer-slots 1 \
    --request-count 4 --height 512 --width 512 --num-frames 16 \
    --num-inference-steps 8 --seed 42'
```

此 A/B 只比较相邻 queued 实现，不是 static-vs-queued，也不是与理想流水线的比较；三次重复支持该优化方向有收益，但仍不足以给出跨环境的稳定性能承诺。Rank 1 的 `transport_progress` 在一个 rank-local 非 profiler 样本中仍超过 1,000 次，说明 Engine/Worker 仍有大量无进展轮询；继续降低这些空轮询和残余 stage idle 是下一项工作。

## 14. 2026-09-28：控制面收缩与自主推进实验

### 保留的控制面改动

提交 `e922fc20c` 进一步减少 PP=2、DP=1、TP/SP/CFG=1 的 Worker 控制同步：

- `submit_pipeline_batch()`、`authorize_pipeline_batch()`、grant start 和 batch release-readiness 由 Executor 收齐两个 rank 的本地 RPC 结果；grant start 仍有 30 s 超时与失败关闭。其他拓扑保留原有 all-rank 协议。
- 下一轮 rank-local progress snapshot 顺带检查已有 transfer offer 的 sender ticket 和 receiver credit，Worker 实际保留 receive credit；只有两个 endpoint 均报告 ready 后，Executor 才发 validated grant。新生成的 offer 在下一轮检查。生产热路径因此不再为每个 offer 单独发 readiness RPC。
- stage 0 在 `edge_buffer_slots > 1` 时可在同一次 progress 中填满可用的 activation send credit，同时保留 FIFO 与 ticket ownership。Engine 仍需等两 rank 的 progress RPC 都返回，此改动没有实现独立的异步 1F1B。
- 每轮 progress 的高频 INFO 日志降为 DEBUG，并仅每 100 个 scheduler round 记录一次摘要。

本地与远端 focused CPU 测试均为 272 passed；Ruff 与 `git diff --check` 通过。以下是相同模型、4 请求、512x512x16、8 steps、seed 42 的单次无 profiler 结果，不作为稳定性能提升结论：

| 提交与配置 | Makespan (ms) | 吞吐 (req/s) | Mean latency (ms) | P50 (ms) |
| --- | ---: | ---: | ---: | ---: |
| `e922fc20c`, slots=1, inflight=2 | 5873 | 0.681 | 4266 | 3372 |
| `e922fc20c`, slots=2, inflight=4 | 5362 | 0.746 | 4368 | 4043 |

两次四个 decoded SHA-256 均与第 13 节列出的对应请求完全一致。较深的队列在本次样本中缩短 makespan，但改变了 admission 深度和请求完成顺序，且均未做重复测量；不能把这 511 ms 差异归因于单一代码改动。结果与 Worker events 已下载到本地 `artifacts/wan22-queued-control-e922fc20-slots1/` 和 `artifacts/wan22-queued-control-e922fc20-slots2/`。后者仍约有 2,200 个 scheduler round，说明控制面轮询和 rank-wide progress barrier 仍在。

### 已撤回的 Worker 自主轮询

提交 `8765ab326` 曾让各 Worker 在两次 RPC 之间自主调用本地 `progress_pipeline_transfers()`，Engine 只轮询缓冲的 metadata；`188846028` 随后尝试用 stage CUDA event 阻止本地 forward 过早提交。这两个实验都完成了全部请求，输出 SHA-256 与保留版本一致，但性能明显退化：

| 实验 | Makespan (ms) | 吞吐 (req/s) | 末两个 `post_decode()` 耗时 (ms) |
| --- | ---: | ---: | ---: |
| 自主轮询 `8765ab326` | 27539 | 0.145 | 11261、12069 |
| 加入 CUDA completion gate `188846028` | 27144 | 0.147 | 12235、10721 |

对比之下，同配置的 `e922fc20c` makespan 为 5362 ms。无 profiler Worker event 显示，自主轮询版本的 32 次 stage-0 forward 在约 2.8 s 内完成；长尾发生在最终 `post_decode()`，不是 denoise forward 或 P2P offer 发布。轻量 marker 进一步确认慢在 `post_decode()` 调用内，而不是结果 RPC 的 SHM 封装。`post_decode()` 包含 latent materialization 和 VAE decode；尚未单独证明是 GPU 排队、内存压力、VAE 内部同步还是线程竞争。CUDA completion gate 没有解决退化，因此不能把原因归结为缺少该 gate。

这两项自主推进改动已通过 `d22cfbda`、`db977e78` 撤回，保留 `e922fc20c` 的控制面改动。诊断目录已下载到 `artifacts/wan22-queued-autonomous-8765ab32-slots2/`、`artifacts/wan22-queued-autonomous-diagnostic-8765ab32/` 和 `artifacts/wan22-queued-device-gate-18884602/`；均是无 profiler 运行。

### 后续架构方向

不应再把 `WorkerProc` 的 RPC 队列超时当作 stage clock。下一版需要真正的 per-rank progress/event 通道：Worker StageEngine 在自身的 bounded device-work、edge credit 和接收事件满足条件时推进；Executor 按 rank 异步收取 metadata，并只在 transfer 两端确认、完整 step commit 和 retirement 处聚合。Engine 应在 event 到达时处理状态转换，而不是每几毫秒重跑一次 `scheduler.schedule()`。同时必须给最终 decode 独立的可观测资源边界，明确它与尚未退休的 DiT device work、CUDA stream、内存预算和结果输出的关系，再决定是否可与 stage progress 并行。验收必须同时要求四请求输出一致、无长尾 VAE 回退、两 rank 的 stage 实际重叠以及端到端吞吐收益；单纯减少 RPC/collective 次数不够。

## 15. 2026-09-28：跳过已持有请求的空 Scheduler 快照

提交 `4cd49a28a` 让 `StepScheduler` 只读判断是否还有未被 queued batch 持有的 running request、可接纳的 waiting request 或待清理的 finished ID。若当前请求均由 retained batch 持有且 queued 容量已满，Engine 直接处理共享 transport/event snapshot 和 retained batch，不再反复调用 `schedule()`、复制 cached descriptors、递增无实际 admission 的 scheduler step ID。batch 退休后，下一轮恢复正常调度；等待队列指标仍从 Scheduler 当前状态读取。Worker progress、P2P grant、最终 decode 和 retirement 时序均未改变。

本地相关 CPU 测试 377 passed；远端新增路径的定向 CPU 测试 59 passed；Ruff 和 `git diff --check` 通过。首次两卡预约因 GPU 0 被另一用户占用，排队超过 5 分钟后取消。两卡随后可用时，通过 `canhazgpu` 在远端 clean checkout `4d87f547` 上运行一次无 profiler 固定 workload：`max_num_seqs=2`、`max_inflight_batches=4`、`edge_buffer_slots=2`、4 请求、512x512x16、8 steps、seeds 42-45。四个请求全部完成，四个 decoded SHA-256 与 `e922fc20c` 对应请求完全一致。

| 同配置单次运行 | Makespan (ms) | 吞吐 (req/s) | Mean latency (ms) | P50 (ms) | 最后一个 batch ID |
| --- | ---: | ---: | ---: | ---: | --- |
| `e922fc20c` | 5362 | 0.746 | 4368 | 4043 | `pp-31-767` |
| `4d87f547` | 5641 | 0.709 | 4630 | 4287 | `pp-31-30` |

batch ID 的末尾字段来自 scheduler step ID，说明空 Scheduler 快照显著减少；但 rank-0 Worker event 文件反而从 3169 条增加到 3434 条，transport progress RPC 仍频繁发生。本次没有测到端到端性能改善，单次样本间的 279 ms 差异也不足以证明稳定回退。该改动主要清理 Engine 的重复调度开销；下一项真正影响流水线空泡的工作仍是按 rank 独立推进 progress、降低空 transport polling，并限制最终 decode 与其他 device work 的资源竞争。新结果和 Worker events 已下载到本地 `artifacts/wan22-queued-scheduler-skip-4d87f547/`。
