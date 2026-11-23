# MiniCPM5-2B：天数 BI-V150 推理优化

本分支发布 FlagOS S2 赛道二的本地达标工程实现。基于上游 `flagos-2026-s2` 的
[`13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`](https://github.com/flagos-ai/vllm-plugin-FL/commit/13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0)，
7 个运行时文件与历史本地 checkpoint `2abbea017e7c3575d715105d3f1819e9d9dff623` 逐字节一致。
该历史 SHA 是原开发仓库的版本标识，本次整理发布产生新的 Git commit，不冒称保留了原提交历史。

## 实现与边界

| 实现 | 代码 | 已记录结果 |
|---|---|---|
| 采样程序并行度从 1×SM 调至 4×SM | [iluvatar_sampler.py](../../vllm_fl/patches/iluvatar_sampler.py) | B64：3.91 → 1.32 ms |
| Prefill query tiling 与 KV 复用 | [iluvatar_prefill.py](../../vllm_fl/patches/iluvatar_prefill.py) | 16K KV attention：75.1 → 29.2 ms |
| 融合分页 KV 读取与两请求打包 | [iluvatar_attention.py](../../vllm_fl/patches/iluvatar_attention.py) | 4K/256 输出短 A/B：+14.52% / +11.33% |
| 分段分页 decode、捕获流工作区预分配与有界复用 | [iluvatar_decode_pages.py](../../vllm_fl/patches/iluvatar_decode_pages.py) | B64/16K attention graph replay：3.50 → 2.21 ms |
| SQLite 仅反射目标配置表 | [iluvatar_config_cache.py](../../vllm_fl/patches/iluvatar_config_cache.py) | 配置初始化：1562 → 12.7 ms |
| 有界 1D int32 索引共享运行时形状 kernel，保留回退 | [iluvatar_index1d.py](../../vllm_fl/patches/iluvatar_index1d.py) | 4 个新形状累计冷启动：443.8 → 1.7 ms |

各行来自不同阶段和负载，局部收益不相乘，短 A/B 不替代下述 1024 输出场景成绩。
入口注册在 [`vllm_fl/__init__.py`](../../vllm_fl/__init__.py)。BF16、16Q/2KV、D128、block16
及捕获阶段等限制由实现检查，超出适用范围保留原路径。

分页 decode 的 attention 核心适配自 [xy200303/FlagGems @ 9c06143c](https://github.com/xy200303/FlagGems/blob/9c06143c/src/flag_gems/fused/paged_decode_attention.py)
（Apache-2.0）；本地工作包括路由、元数据映射、空行安全、stream-owned scratch 和图捕获集成。
Prefill 复用平台 vendor kernel。源码版权及[第三方来源记录](references/provenance.json)保留，
不将这些上游核心表述为从零自研。

## 模型结果

BI-V150 单卡，MiniCPM5-2B，输入/输出/并发/请求数分别如下。

| 工作负载 | Total tokens/s | 对公布基线倍率 | Mean TTFT | 对公布基线变化 |
|---|---:|---:|---:|---:|
| [4096, 1024, 64, 256] | 4241.91 | 2.09× | 4294.85 ms | -62.9% |
| [16384, 1024, 64, 128] | 3649.31 | 3.99× | 137478.55 ms | -77.1% |

Total tokens/s 包含输入和输出 token，不是纯 decode 速度。每场景四轮，排除首轮后取后三轮均值；
[原始 CSV](throughput-raw.csv)、[原始审计](throughput-audit.json)及[资格核对记录](qualification.json)随代码保留。
这是本地测量相对公布基线的比较，包含指定 FlagTree 0.7 环境；编译器环境升级不单独计为算法贡献。

独立完整 MATH-500 Level 3 复测为 103/105（98.10%）。首测 99/105、原性能记录遗漏 index1d
导致的两条身份审计错误均保留，未追溯改写为通过；精度复测七文件 SHA 对齐不消除原性能记录缺口。
本分支不是主办方验收或获奖证明，历史 manifest 的 `official_submission_status` 也不是实时提交状态。
发布时仅核对源码并运行 CPU 校验，没有重跑 GPU 成绩。未采用的 W8A8、稀疏 attention、vendor paged
及小 fill 实验不包含在本分支中。

硬件剖析：模型内 BF16 GEMM 为 71–73 TFLOPS，达到同卡实测 95 TFLOPS 上限的 75%–77%，
这个比值不是理论 MFU。16K KV Prefill 的 `ixkn` 记录为 LSU Shared 利用率 85.75%、occupancy
24.87%，用于定位共享访存限制，不将其称为 HBM MBU。

## 环境与复现入口

使用与历史结果匹配的官方 BI-V150 Linux 镜像，保留原厂 COREX、ixformer 和原装 vLLM 扩展。

| 依赖 | 已验证版本 |
|---|---|
| PyTorch | 2.10.0+corex.4.5.0 |
| vLLM | 0.24.0+empty |
| vllm-plugin-fl 原装 metadata | 0.3.0rc0+g13eb9be69 |
| FlagGems | 5.3.5 / a7620cc191a0b42e040194622c5758b22a7a25dc，未改动 |
| FlagTree | 0.7.0+iluvatar3.6 |
| ixformer | 0.7.0+corex.4.5.0 |
| EvalScope | 1.11.1 |
| SQLAlchemy | 2.0.48，要求 2.0.x |

下载源码并核对七文件（仅标准库，无 GPU 初始化）：

```bash
git clone --branch iluvatar-minicpm5-2b-optimization https://github.com/dandan20031031/vllm-plugin-FL.git
cd vllm-plugin-FL
python3 tools/iluvatar/verify_checkpoint.py
python3 tools/iluvatar/check_overlay_cpu.py
```

在匹配镜像中先停服务，再使用 overlay 安装；它保留原装 metadata 和原厂扩展，
检查已安装插件、FlagGems 与原生 attention 文件的版本和指纹，默认 dry-run。
安装器由原工程工具迁移，仅调整 fork 内的源文件和 manifest 路径；保留目标环境与七文件校验、备份及回退。
不要用重新 pip 安装 fork 来替代此已验证部署方式。

```bash
python3 tools/iluvatar/install_overlay.py
python3 tools/iluvatar/install_overlay.py --apply
# 使用上一步实际打印的备份目录回退：
python3 tools/iluvatar/install_overlay.py --rollback /exact/printed/backup
```

模型 `/workspace/MiniCPM5-2B` 和数据集 `/workspace/evalscope-datasets/math_500` 由镜像提供，
本仓库不分发权重或原始评测题。启动与性能评测参数保持原口径：

```bash
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072

# 在另一终端，服务就绪后运行：
python3 benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

精度口径为 EvalScope / MATH-500 Level 3、batch 8、temperature 1.0、top_p 0.95、
max_tokens 32768；每次使用新 work-dir，检查完整 105 条预测和评分。
[manifest](manifest.json)记录历史版本、文件 SHA 与未解决的审计边界。
