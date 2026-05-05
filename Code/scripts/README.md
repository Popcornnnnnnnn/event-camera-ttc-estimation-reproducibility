# Scripts

当前活动脚本只保留下一阶段仍可复用的通用工具。

## 当前保留

- `data/`
  - `build_dataset_manifest.py`
  - `build_formal_sequence_artifacts.py`
  - `run_confidence_smoke.py`
  - `run_compare_bucket_report.py`
  - `run_compare_regression_report.py`
  - `run_formal_loader_smoke.py`
  - `run_lts_smoke.py`
  - `run_local_cue_v2_smoke.py`
  - `run_event_expansion_probe.py`
  - `run_sparsity_aware_lts_compare.py`
  - `run_bbox_looming_failure_diagnostics.py`
  - `run_sparsity_aware_lts_v2_smoke.py`
  - `run_sparsity_aware_lts_eval.py`
  - `run_small_sample_smoke.py`
- `download/`
  - `download_evtcc_asset.py`
  - `download_evtcc_ccrs1_batch.py`
  - `watch_evtcc_ccrs1_progress.py`
- `preview/`
  - `preview_event_windows.py`
  - `preview_slider_bag.py`

## 已归档

以下 Phase 1 基线复现脚本已移入：

- `Archive/Phase1Baseline/LegacyCode/Code/scripts/baselines/`
- `Archive/Phase1Baseline/LegacyCode/Code/scripts/download/`
- `Archive/Phase1Baseline/LegacyCode/Code/scripts/reporting/`

对应的正式实验、本地工作区和 third-party 产物则已转入：

- `Archive/Phase1Baseline/ArtifactsLocal/`

## 当前原则

- 活动区只保留通用工具。
- 旧 baseline 脚本不再作为默认主入口。
- 如需回看 Phase 1 实现，请直接进入 `Archive/Phase1Baseline/`。

## 新增 Phase 2 接口脚本

- `data/build_formal_sequence_artifacts.py`
  - 作用：从 `Dataset-Full/<sequence>/` 生成 `Code/DatasetFormal/<sequence>/` 下的统一中间层文件
  - 当前输出：
    - `SequenceMeta.json`
    - `Protocol.json`
    - `Calib.json`
    - `bbox.csv`
    - `gt_ttc.csv`
    - `IntervalIndex.csv`
  - 默认行为：
    - 若序列输出已完整，重跑时自动跳过
    - 若发现中断留下的不完整文件集，只重建该序列
    - 若需强制重建，使用 `--overwrite-existing`

- `data/run_formal_loader_smoke.py`
  - 作用：验证统一读取层能否正确读取 `SequenceMeta / Protocol / Calib / IntervalSample / EventSlice`
  - 依赖读取模块：
    - `Code/evttc/formal_loader.py`

- `data/run_lts_smoke.py`
  - 作用：验证 `NativeSampleAdapter -> LtsV1 -> LocalStatsV1` 的最小表示链路
  - 依赖模块：
    - `Code/evttc/native_adapter.py`
    - `Code/evttc/representations.py`

- `data/run_confidence_smoke.py`
  - 作用：验证 `LocalStatsV1 -> LocalConfidenceV1` 的第一版局部可信度链路
  - 依赖模块：
    - `Code/evttc/confidence.py`
    - `Code/evttc/representations.py`

- `data/run_local_cue_v2_smoke.py`
  - 作用：验证 `LocalConfidenceV1 -> LocalTtcCueV2` 的第二版局部 cue 构建链路
  - 依赖模块：
    - `Code/evttc/sparsity_aware_lts.py`
    - `Code/evttc/confidence.py`
    - `Code/evttc/representations.py`

- `data/run_sparsity_aware_lts_v2_smoke.py`
  - 作用：在单一样本上直接比较 `predict_ttc_v1()` 与 `predict_ttc_v2()` 的输出差异
  - 依赖模块：
    - `Code/evttc/sparsity_aware_lts.py`
    - `Code/evttc/confidence.py`
    - `Code/evttc/representations.py`

- `data/run_compare_regression_report.py`
  - 作用：从 `CompareIntervalMetrics.csv` 中提取回归最大和改善最大的样本
  - 依赖模块：
    - Python stdlib `csv`

- `data/run_compare_bucket_report.py`
  - 作用：将 `CompareIntervalMetrics.csv` 分桶为 `GUARD_FALSE_REJECT / FALLBACK_WIN_BIG / LOW_SUPPORT` 等可执行诊断类别，并输出 `BucketedCompare.csv + BucketSummary.md`
  - 依赖模块：
    - Python stdlib `csv`

- `data/run_sparsity_aware_lts_eval.py`
  - 作用：运行第一版 `sparsity-aware LTS` 完整评测闭环，输出 `Config / IntervalMetrics / Summary`
  - 当前支持：
    - `HeuristicCueV1`
    - `BBoxLoomingV1..V13`
    - `BBoxLoomingV5` 默认启用 `candidate_count / max_fit_points / IQR` 质量门控
    - `BBoxLoomingV6` 对低置信样本使用 quality-aware adaptive quantile，不回退到旧 LTS
    - `BBoxLoomingV7` 在 V6 上加入短候选 spike guard
    - `BBoxLoomingV8` 在 V7 上加入当前 bbox 面积崩塌 guard
    - `BBoxLoomingV9` 在 V8 上加入历史 bbox 单帧面积崩塌 guard
    - `BBoxLoomingV10` 将 V9 的历史 bbox guard 限制为低置信候选触发
    - `BBoxLoomingV11` 在 V10 上拒判剩余低置信候选，作为高置信输出分支
    - `BBoxLoomingV12` 在 V11 高置信分支上加入 phase-aware quantile：早期高离散用 `Q50`，晚期低离散用 `Q10`
    - `BBoxLoomingV13` 保留 V12 的 late `Q10`，额外对稳定慢增长高置信样本用 `Q10`、对早/中期增长型高置信样本用 `Q50`
  - 依赖模块：
    - `Code/evttc/sparsity_aware_lts.py`
    - `Code/evttc/confidence.py`
    - `Code/evttc/representations.py`

- `data/run_bbox_looming_failure_diagnostics.py`
  - 作用：诊断 `BBoxLoomingV4` 高误差 interval，输出逐样本诊断表、候选 TTC 明细和 SVG 可视化
  - 当前输出：
    - `IntervalDiagnostics.csv`
    - `SequenceDiagnostics.csv`
    - `WorstIntervals.csv`
    - `WorstIntervalCandidates.csv`
    - `plots/*.svg`

- `data/run_bbox_failure_type_report.py`
  - 作用：对 `BBoxLoomingV10` 等结果做 failure type 标注，输出剩余高误差类型、status family 汇总和拒判场景表
  - 当前默认输入：`BBoxLoomingV13_20260427_212537`
  - 当前输出：
    - `IntervalFailureTypes.csv`
    - `HighErrorIntervals.csv`
    - `FailureTypeSummary.csv`
    - `StatusFamilySummary.csv`
    - `SequenceFailureTypeSummary.csv`
    - `RejectionScenarios.csv`
  - 依赖模块：
    - `Code/evttc/sparsity_aware_lts.py`
    - `Code/evttc/native_adapter.py`

- `data/run_bbox_high_conf_geometry_report.py`
  - 作用：复算高置信 `BBoxLooming` 样本的多窗口候选 TTC 分位数，诊断 fixed `Q25` 对 apparent geometry over/under 的影响
  - 当前默认输入：`BBoxLoomingV13_20260427_212537` 与对应 `BBoxFailureTypeReport_20260427_212614`
  - 当前输出：
    - `IntervalCandidateQuantiles.csv`
    - `QuantileErrorSummary.csv`
    - `RuleProbeSummary.csv`
    - `FeatureGroupSummary.csv`
    - `WorstHighConfIntervals.csv`
    - `HighConfGeometryCandidates.csv`
  - 依赖模块：
    - `Code/evttc/sparsity_aware_lts.py`
    - `Code/evttc/native_adapter.py`

- `data/render_thesis_bbox_figures.py`
  - 作用：将论文候选样本渲染为轻量 SVG 图，包含 bbox 面积历史、当前预测/GT 和可选候选分位数误差面板
  - 当前默认输入：
    - `BBoxLoomingV13_20260427_212537`
    - `BBoxFailureTypeReport_20260427_212614`
    - `BBoxHighConfGeometryReport_20260427_212710`
  - 当前默认输出示例：
    - `Code/Experiments/SparsityAwareLTSDiagnostics/ThesisBBoxFigures_20260427_213754/`
  - 依赖模块：
    - `Code/evttc/native_adapter.py`

- `data/run_event_expansion_probe.py`
  - 作用：验证 naive event-cloud expansion cue 是否能补 `BBoxLooming` 的剩余失败样本
  - 当前输出：
    - `EventExpansionMetrics.csv`
    - `EventExpansionCandidates.csv`
    - `Summary.json`
    - `Summary.md`
  - 依赖模块：
    - `Code/evttc/native_adapter.py`

- `data/prepare_formal_batch.py`
  - 作用：批量整理 `Dataset-Full` 序列并生成可恢复的 `DatasetFormal` artifacts
  - 当前实现：
    - 自动抬平 `DCV/` 嵌套目录
    - 可清理 `GVT/` 与浏览器下载残留包
    - 已完整的 formal artifacts 自动跳过
    - 单条序列失败只记录，不阻断整批
  - 当前输出：
    - `Code/Derived/DataIngestion/<run_tag>/SequenceInventory.csv`
    - `Code/Derived/DataIngestion/<run_tag>/Summary.json`

- `data/export_ml_baseline_dataset.py`
  - 作用：将 `EvTTC-Formal` 样本导出为轻量学习 baseline 可直接使用的 `npz` 数据集
  - 当前输出：
    - 每条序列一个 `dataset.npz`
    - 一个聚合 `Summary.json`
    - 一个序列级 `ExportStatus.csv`
  - 当前恢复行为：
    - 默认跳过已有且可读的 `dataset.npz`
    - 使用 `--overwrite-existing` 可强制重导
    - 单条序列导出失败只记录，不阻断整批
  - 当前导出内容：
    - `image_nchw`
    - `stats`
    - `metadata`
    - `gt_ttc_s`
    - `gt_log_ttc_s`
    - `sample_id / sequence_id / interval_idx`
  - 当前图像表示：
    - `lts_v1`
    - `polarity_split_v1`
    - `multi_tau_polarity_split_v1`
  - `multi_tau_polarity_split_v1` 当前默认 `tau`：
    - `10ms`
    - `20ms`
    - `40ms`
    - 每个 `tau` 各导出一组 `6` 通道 polarity-split 图像，总计 `18` 通道
  - 依赖模块：
    - `Code/evttc/ml_baseline.py`
    - `Code/evttc/formal_loader.py`
    - `Code/evttc/native_adapter.py`
    - `Code/evttc/representations.py`

- `data/run_ml_baseline_train.py`
  - 作用：在导出的 `MlBaselineV1` 数据集上运行按 sequence 分组的最小学习 baseline 训练/评测
  - 当前实现：
    - `leave-one-sequence-out`
    - `ridge regression`、`tiny MLP` 或 `tiny CNN`
    - `ridge / MLP`:
      - `stats+metadata`
      - `pooled image + stats + metadata`
    - `CNN`:
      - `image_only`
      - `image + stats + metadata`
      - `image_late_fusion_stats_metadata`
      - `image_residual_stats_metadata`
    - `tiny MLP` 当前为 numpy 版 `ReLU MLP + Adam + early stopping`
    - `tiny CNN` 当前为 numpy 版两层卷积网络：
      - 输入先做 `2x` 平均降采样
      - `conv -> relu -> avgpool -> conv -> relu -> avgpool -> global avg pool`
      - 再接一个小型 `MLP head`
    - 当前附加融合原型：
      - `image_late_fusion_stats_metadata`
        - 图像分支与 stats/meta 分支各自编码后在末端融合
      - `image_residual_stats_metadata`
        - 以 `image_only` 为主分支，stats/meta 只学习残差校正
    - 当前 CNN 默认候选超参：
      - `channels=[8,12]`
      - `head_hidden_dim=24`
      - `learning_rate=5e-4`
      - `weight_decay=3e-4`
  - 当前输出：
    - 每个 test sequence 一个 `IntervalMetrics.csv + Summary`
    - `MLP / CNN` 额外输出每个 fold 的 `TrainHistory.csv`
    - 一个 overall `Config / IntervalMetrics / Summary`
    - 训练中持续更新 `RunStatus.json`
      - 当前 fold / epoch / batch 心跳
      - epoch 级 `train_loss`、`val_loss`、`best_epoch`
      - `running`、`interrupted`、`failed`、`complete` 状态
  - 当前恢复行为：
    - `--run-name` 指定固定输出目录
    - 默认跳过已有完整 fold
    - `--overwrite-existing-folds` 可强制重训 fold
    - `KeyboardInterrupt` 会写入 `interrupted`，避免残留假 `running`
  - 依赖模块：
    - `numpy`
    - 导出的 `Code/Derived/MlBaselineV1/<tag>/<sequence>/dataset.npz`
  - 当前主线建议：
    - 优先比较同一 `TinyCnnImageOnlyV1` 在不同输入表示上的表现
    - 先做 `polarity-split`
    - 再做 `multi-tau + polarity-split`

- `data/run_ml_baseline_train_torch.py`
  - 作用：在导出的 `MlBaselineV1` 数据集上运行 PyTorch 版 tiny CNN grouped baseline，优先使用 Apple MPS 加速
  - 当前实现：
    - `leave-one-sequence-out`
    - `image_only`
    - `image_stats_metadata`
    - `image_bbox_aux`
      - 需要 `--aux-metrics-csv`
      - 当前用于接入 `BBoxLoomingV7` 的 `log(ttc_est)`、valid flag、confidence
    - `image_stats_metadata_bbox_aux`
    - `--prediction-mode direct`
      - 默认模式，直接预测 `log(TTC)` 或 `TTC`
    - `--prediction-mode residual_to_aux`
      - 仅支持 `target=log_ttc` 与 bbox aux feature mode
      - 训练目标为 `log(gt_ttc) - log(aux_ttc)`，推理时加回 aux bbox log-TTC
      - 当前 `TorchTinyCnnBBoxResidualV1` 全量结果未优于 direct bbox aux
    - 网络结构对齐 NumPy `TinyCnnImageOnlyV1` 主干：
      - 输入先做 `2x` 平均降采样
      - `conv -> relu -> avgpool -> conv -> relu -> avgpool -> global avg pool`
      - 再接一个小型 `MLP head`
    - 默认设备：
      - `--device auto` 优先选择 `mps`
    - 当前默认候选超参：
      - `channels=[8,12]`
      - `head_hidden_dim=32`
      - `batch_size=64`
      - `learning_rate=5e-4`
      - `weight_decay=3e-4`
  - 当前输出：
    - 输出根目录默认 `Code/Experiments/MlBaselineTorch/`
    - 每个 test sequence 一个 `IntervalMetrics.csv + Summary + TrainHistory.csv + Model.pt`
    - 一个 overall `Config / IntervalMetrics / Summary`
    - 训练中持续更新 `RunStatus.json`
  - 当前恢复行为：
    - `--run-name` 指定固定输出目录
    - 默认跳过已有完整 fold
    - `--overwrite-existing-folds` 可强制重训 fold
  - 依赖模块：
    - `torch`
    - `numpy`
    - 导出的 `Code/Derived/MlBaselineV1/<tag>/<sequence>/dataset.npz`
    - 可选的传统方法 `IntervalMetrics.csv`

- `download/download_evtcc_ccrs1_batch.py`
  - 作用：批量下载缺失的 `CCRs-1` 全序列原始资产到 `Dataset-Full/`
  - 当前实现：
    - 大文件：
      - `gdown --fuzzy --continue`
    - `leftlabel`:
      - Google Drive mobile folder 页面解析
      - `curl + retry` 逐个 JSON 下载
    - 自动完整性检查：
      - 大文件最小体积
      - `leftlabel` 必须达到预期 json 数量

- `download/watch_evtcc_ccrs1_progress.py`
  - 作用：在终端里实时展示 `CCRs-1` 批量下载进度
  - 当前输出：
    - downloader 进程状态
    - 每条序列的：
      - `HDF5 / MP4 / BAG / GT / TTC / LeftLabel`
    - overall 完成百分比
  - 常用方式：
    - 单次快照：
      - `Code/.venv/bin/python Code/scripts/download/watch_evtcc_ccrs1_progress.py --once`
    - 持续刷新：
      - `Code/.venv/bin/python Code/scripts/download/watch_evtcc_ccrs1_progress.py`
