# SQuaDE

冻结 **DINOv2 ViT-g** + 两组各三个专家头，检测 AI 生成图。干净图在第 27 个 block
提前退出，省约 20% 前向深度。为 NTIRE 2026 Robust AIGI Detection 训练。

```
输入 → DINOv2 ViT-g(冻结, 40 blocks)
         │
         ├─ 跑到 L27,取 L14/L21/L27 → 浅组三专家 均匀平均 → 分数
         │                          → 门(读同样的浅层特征)
         │       门判"干净" → 就在这里输出,后面 13 个 block 根本不跑
         │
         └─ 门判"退化" → 从 L27 的 hidden 续跑到 L37,取 L26/L33/L37 → 深组
```

---

## 快速上手

```bash
git clone https://github.com/kelvinchua1099-alt/SQuaDE && cd SQuaDE
pip install -r requirements.txt
```

DINOv2 权重会在首次运行时自动下载，不需要申请权限。

```bash
# 拉权重(36 MB,只含专家头与门,不含骨干)
huggingface-cli download kelvinchua/squade-vitg --local-dir ckpt

# 跑一个目录
python Inference.py --dir /path/to/images \
  --shallow ckpt/vg_shallow_full \
  --deep    ckpt/vg_deep_full \
  --gate    ckpt/gate_full.pt \
  --out preds.csv

# 单张图
python Inference.py --image a.jpg --shallow ckpt/vg_shallow_full \
  --deep ckpt/vg_deep_full --gate ckpt/gate_full.pt
```

## 输出

```
image                    score     label  route    experts       votes              ms_per_img
002d7df53e3ae55af5.jpg   1.000000  1      deep     L26/L33/L37   1.000|1.000|0.955  684
0053097bfa680600.jpg     0.000000  0      shallow  L14/L21/L27   1.000|0.000|0.000  684
```

| 列 | 含义 |
|---|---|
| `score` | 0~1，越大越可能是 AI 生成 |
| `label` | `score > 0.5` 时为 1（假） |
| `route` | 走了哪一组专家 |
| `experts` | 该组的三个抽头层 |
| `votes` | 组内三个专家各自的概率 |
| `vote_spread` | 三票的最大差。**大 = 这张图在决策边界上**，最值得人工复核 |
| `depth_used` | 实际跑了多少个 block（27 或 37，总共 40） |
| `gate_logit` | 门的输出，≤0 判「干净」走浅组 |
| `ms_per_img` | 每张图耗时 |

## 常用参数

```bash
--batch-size 8        # 24 GB 显存下可到 16
--no-early-exit       # 两组都算完再按门选。精度相同,用来核对提前退出没改变结果
--device cpu          # 慢很多,应急用
--crop-size 504       # 默认值,DINOv2 patch=14 的整数倍,别改
```

---

## 输入尺寸怎么处理

推理与训练用**同一套规则**。不一致等于给测试集加一层没记录在案的域偏移，
而且不会报错。

```
短边 >= 512   中心裁 512x512,一次重采样都不做
              —— 保住原生高频(生成器指纹住在那里),同时消除"缩放倍率"这条捷径
短边 <  512   先裁成正方,再用随机挑的一个内核上采样到 512
              内核池 [BILINEAR, BICUBIC, BOX, LANCZOS],按图名确定性挑
              —— 固定一个内核的话,"带哪种插值痕迹"会变成与内容无关的伪线索
```

之后送进骨干时中心裁到 **504**（patch=14 的整数倍），同样不 resize。
以上都由 `Inference.py` 自动完成，不用自己预处理。

---

## 速度

24 GB 显存、batch 8、504×504：约 **1.5 图/秒**（684 ms/图）。

干净图在 L27 停止，省掉 L28~L37 共 13/40 个 block。官方 val 上门把约 60% 的图
判为干净，实测省 **19.7%** 前向深度。

⚠️ 在**原生低分辨率**来源上门会失效——它学到的其实是「图糊不糊」而不是「有没有
施加过退化算子」。实测 200×200 上采样的图里 82% 被判成退化，省下的算力掉到 5.8%。
精度不受影响。

---

## 成绩

| 数据 | 全量 | clean | robust |
|---|---|---|---|
| 官方 val (10,000) | 0.9713 | 0.9820 | 0.9583 |
| 官方 val_hard (2,500) | 0.8818 | 0.9488 | 0.7981 |
| DALL·E 3 vs COCO (17,434) | 0.9943 | 0.9960 | 0.9925 |

`robust` = 只在退化图上算，是竞赛主指标。模型从未见过这三份数据。

---

## 其余文档

- `CLAUDE.md` — 训练流水线、八条会静默出错的纪律
- `docs/probe_protocol.md` — 层级探针的运行设定
- `docs/design_experiments.md` — 每个设计决策背后的实测

MIT。DINOv2 权重另有其许可条款。
