# Micro-batch Image Classifier

为视觉部署场景（多台相机）提供的**限时图片分类**服务：FastAPI + CPU 版 PyTorch，
单进程微批（micro-batch）推理。提交图片时给出从受理起算的截止毫秒数；
超时或被取消的帧不会产出结果。每台相机可配置一份 10 类参考分布，
服务据最近 20 个有效完成单监测分类分布是否突变。

## 快速开始

```bash
docker compose up --build
# 服务: http://localhost:8000   交互文档: http://localhost:8000/docs
```

本地开发（无需 Docker）：

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --port 8000
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/predict` | 提交图片（multipart `file`）+ 表单 `deadline_ms`（从受理起算）+ 可选 `camera_id`。返回 `order_id`。满额返回 **429** |
| GET | `/orders/{order_id}` | 按单号查询状态与结果 |
| POST | `/orders/{order_id}/cancel` | 按单号取消；已完成返回 **409** |
| GET | `/stats` | 容量/状态计数（演示与观测用） |
| GET | `/health` | 存活探针 |
| PUT | `/cameras/{camera_id}/reference` | 设置/更新该相机 10 类参考百分比 |
| GET | `/cameras/{camera_id}/reference` | 读取当前参考值（含版本号） |
| DELETE | `/cameras/{camera_id}/reference` | 删除参考值及其窗口 |
| GET | `/cameras/{camera_id}/distribution` | 最近 20 个有效完成单的类别占比、TVD 与告警判定 |
| GET | `/cameras` | 所有已配置相机的当前分布摘要 |

示例：

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@samples/sample_gradient.png" -F "deadline_ms=5000"
# {"order_id":"7f6770829f36","status":"queued",...}

curl http://localhost:8000/orders/7f6770829f36
# {"status":"completed","result":{"predicted_class":1,"scores":[...10 个分数...]},...}

curl -X POST http://localhost:8000/orders/7f6770829f36/cancel
```

单号状态机：`queued → processing → completed`；`queued/processing → cancelled|expired`。
`result.scores` 为 10 类 softmax 分数，与提交单号一一对应。

## 按相机监测（视觉值班）

### 1. 配置参考分布

```bash
curl -X PUT http://localhost:8000/cameras/cam-front/reference \
  -H 'content-type: application/json' \
  -d '{"percentages":[10,10,10,10,10,10,10,10,10,10]}'
# {"camera_id":"cam-front","version":1,"percentages":[...],"updated_at":"...",
#  "distribution":{...}}
```

`percentages` 必须是 **10 个非负整数**（布尔、浮点、字符串都会被拒）且 **合计 100**；
否则返回 **422**，且原参考值保持不变。

### 2. 提交时附相机号

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@samples/sample_gradient.png" -F "deadline_ms=5000" -F "camera_id=cam-front"
```

- 不填 `camera_id`：完全按原方式推理，不进入任何监测窗口（订单视图里两个字段为 `null`）。
- 填了但该相机**尚未配置参考值**：照常受理并交付分类结果，但不参与监测
  （`camera_version` 为 `null`）；之后配置参考值也不会追认这些帧。
- 订单视图新增 `camera_id` / `camera_version` 两个字段，便于追溯。

### 3. 查询占比与告警

```bash
curl http://localhost:8000/cameras/cam-front/distribution
```

```json
{
  "camera_id": "cam-front",
  "reference_version": 1,
  "reference_percentages": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
  "window_limit": 20,
  "sample_count": 20,
  "counts": [0, 20, 0, 0, 0, 0, 0, 0, 0, 0],
  "proportions": [0.0, 1.0, 0.0, ...],
  "percentages": [0.0, 100.0, 0.0, ...],
  "tvd": 0.9,
  "threshold": 0.3,
  "status": "anomaly",
  "anomaly": true,
  "message": "分布异常：总变差距离 0.9 > 0.3"
}
```

- `counts` / `proportions` / `percentages`：窗口内 10 类的计数、占比（0–1）与百分数。
- **总变差距离**：`TVD = 0.5 × Σ|实际占比ᵢ − 参考占比ᵢ|`，参考占比 = 参考百分比 / 100。
- `status`：
  - `insufficient_sample`：有效完成单不足 20（样本不足），此时 `anomaly=false`，
    窗口非空时仍会给出当前 `tvd` 供参考；
  - `anomaly`：`TVD > 0.3`（严格大于；恰为 0.3 不算异常）；
  - `normal`：≥ 20 单且 `TVD ≤ 0.3`。

### 4. 窗口与版本语义

- 窗口只保留**最近 20 个有效完成单**，先进先出。
- 只有「按当前参考版本受理、且成功完成」的单进入窗口：
  **取消、过期、失败**的单一律不计数（未进入模型、结果被丢弃或模型报错）。
- 每次 `PUT` 参考值都会产生**新版本号**，并**立即清空该相机窗口**，
  因此旧的异常告警会马上消失（`status` 变回 `insufficient_sample`）。
- 更新前已受理、更新后才完成的帧：`camera_version` 记录的是受理时的旧版本，
  完成时版本不匹配，**不会进入新窗口**；但其分类结果仍照常通过
  `/orders/{order_id}` 交付。
- `DELETE` 后再重建同一 `camera_id`，版本号继续递增，不会复用旧版本，
  因此删除前受理的帧也无法污染新参考值的窗口。
- 窗口内某类计数为 0 是合法的（例如参考值为 `[100,0,...]` 时）。

监测状态保存在内存中，与订单一样重启即丢失（演示规模）。

## 规则实现（与需求逐条对应）

- **容量 16**：排队中 + 计算中合计最多 16 张，满额提交返回 429。
- **单批 ≤ 4，等批 ≤ 50ms**：模型空闲时最多等 50ms 凑批，凑满 4 张立即开批。
- **最多一批实际推理**：单个 worker 协程 + 单线程执行器，任何时刻只有一批在做前向计算。
- **排队取消/到期**：从队列移除、释放容量，**不进入模型**（`batch_id` 始终为空）。
- **入批后取消/到期**：立即标记（查询即刻可见）、不供结果；**容量占用到该批计算实际结束**，
  期间不会因此另开一批（worker 严格串行）。
- **批内其他图片正常完成**，分数对应原单；批推理与逐张推理误差 ≤ 1e-5（实测约 7.5e-9）。
- **迟到结果不恢复取消/过期单**：结果回写只针对仍处于 `processing` 且未过期的单。
- **慢推理不挡 API**：阻塞式前向计算放在独立线程，事件循环始终可用。
- **监测钩子不改变上述行为**：完成回调仅在订单进入 `completed` 后调用一次；
  回调异常只记日志，不影响推理与结果交付。容量、凑批、取消逻辑保持不变。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAX_CAPACITY` | 16 | 排队+计算中最大图片数 |
| `BATCH_SIZE` | 4 | 单批最大图片数 |
| `BATCH_WAIT_MS` | 50 | 空闲模型凑批等待上限 |
| `INFERENCE_DELAY_MS` | 0 | **演示用**：在推理线程内注入的人为延迟 |
| `MODEL_PATH` | `app/model.pt` | TorchScript 模型路径 |

监测窗口（20）、阈值（0.3）和参考百分比格式（10 类、非负整数、合计 100）由需求固定，
不提供环境变量覆盖，避免值班配置与判定口径漂移。

慢推理演示：`docker compose up` 前把 `docker-compose.yml` 里的 `INFERENCE_DELAY_MS`
改为 `2000`，推理约 2 秒/批，期间提交/查询/取消仍在毫秒级响应（可用 `/stats` 观察），
也可用来观察「更新参考值时仍有旧版本帧在算」的场景。

## 模型与样图

- `app/model.pt`：固定权重 TorchScript 模型（小型 CNN，10 类，输入 64×64 RGB）。
  由 `scripts/make_model.py` 以固定种子生成，权重 SHA-256 指纹见脚本输出，可复现。
- `samples/`：4 张确定性生成的样图（渐变/圆环/条纹/噪声）。

重新生成：

```bash
python scripts/make_model.py --out app/model.pt
python scripts/make_samples.py --out samples
```

### 在 /docs 里操作

打开 <http://localhost:8000/docs>：

1. `PUT /cameras/{camera_id}/reference` → *Try it out* → `camera_id` 填 `cam-front`，
   Request body 填 `{"percentages":[10,10,10,10,10,10,10,10,10,10]}` → *Execute*。
2. `POST /predict` → 选一张 `samples/` 下的图，`deadline_ms` 填 `5000`，
   `camera_id` 填 `cam-front` → *Execute*，拿到 `order_id`。
3. `GET /cameras/{camera_id}/distribution` → `camera_id` 填 `cam-front` → *Execute*，
   查看 `sample_count` / `percentages` / `tvd` / `status`。
4. 再 `PUT` 一次参考值，随后立即 `GET .../distribution`，
   可看到 `sample_count` 归零、`status` 回到 `insufficient_sample`（旧告警已撤下）。

## 测试

```bash
.venv/bin/pytest tests/ -q
```

覆盖：单张完成、批/单张分数一致性（≤1e-5）、凑批窗口、容量 429、排队取消、
入批取消（立即标记+容量仍占用+不另开批）、完成单取消冲突、排队过期、
计算中过期、迟到结果不复活、慢推理下 API 延迟。

`tests/test_monitoring.py` 另覆盖按相机监测：参考值校验（长度/负数/浮点/布尔/和≠100）、
窗口仅保留 20 单、样本不足判定、TVD 严格大于 0.3 的边界（0.3 不算异常）、
更新参考值立即清空窗口并撤下告警、更新前受理的帧完成时照常交付但不进入新窗口、
取消/过期/失败单不计数、删除并重建相机时版本隔离、不填 `camera_id` 时行为不变。
API 层测试用确定性推理桩控制预测类别，其余（容量/凑批/截止/取消）均为真实引擎逻辑。

## 已知限制

- 订单与相机参考值均保存在内存中，重启即丢失；未做持久化与历史订单清理（演示规模）。
- 单进程单 worker；多副本扩展需外加负载均衡与共享状态层，不在本交付范围。
- 监测窗口是最近 20 个完成单的滑动窗口，不做时间衰减或按小时聚合；告警以查询时
  的最新窗口为准，不主动推送。
