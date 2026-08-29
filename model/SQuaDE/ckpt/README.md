---
license: mit
tags:
  - image-classification
  - ai-generated-image-detection
  - deepfake-detection
  - dinov2
library_name: pytorch
---

# SQuaDE ViT-g

冻结 **DINOv2 ViT-g** + 两组各三个专家头，检测 AI 生成图。干净图在第 27 个 block
提前退出，省约 20% 前向深度。

骨干不在本仓库内，运行时从 `facebook/dinov2-giant` 加载。本仓库只有专家头与门（36 MB）。

---

## 快速推理

```bash
git clone https://github.com/<your>/SQuaDE && cd SQuaDE
pip install -r requirements.txt

huggingface-cli download kelvinchua/squade-vitg --local-dir ckpt

python Inference.py --dir /path/to/images \
  --shallow ckpt/vg_shallow_full \
  --deep    ckpt/vg_deep_full \
  --gate    ckpt/gate_full.pt \
  --out preds.csv
```

单张图把 `--dir` 换成 `--image a.jpg`。

### 输出

```
image                 score     label  route    experts       votes              ms_per_img
002d7df53e3ae55af5.jpg 1.000000  1     deep     L26/L33/L37   1.000|1.000|0.955  684
0053097bfa680600.jpg   0.000000  0     shallow  L14/L21/L27   1.000|0.000|0.000  684
```

| 列 | 含义 |
|---|---|
| `score` | 0~1，越大越可能是 AI 生成 |
| `label` | `score > 0.5` 时为 1（假） |
| `route` | 走了哪一组专家 |
| `experts` | 该组的三个抽头层 |
| `votes` | 组内三个专家各自的概率——三票分歧大说明这张图在决策边界上 |
| `depth_used` | 实际跑了多少个 block（27 或 37，总共 40） |
| `gate_logit` | 门的输出，≤0 判「干净」走浅组 |

### 常用参数

```bash
--batch-size 8        # 24 GB 显存下可到 16
--no-early-exit       # 两组都算完再按门选,精度相同,用来核对提前退出没改变结果
--device cpu          # 慢很多,应急用
```

---

## 输入尺寸怎么处理

推理与训练用**同一套规则**，不一致等于给测试集加一层没记录的域偏移：

```
短边 >= 512   中心裁 512x512,一次重采样都不做
              —— 保住原生高频(生成器指纹住在那里),也消除"缩放倍率"这条捷径
短边 <  512   先裁成正方,再用随机挑的一个内核上采样到 512
              内核池 [BILINEAR, BICUBIC, BOX, LANCZOS],按图名确定性挑
              —— 固定一个内核的话,"带哪种插值痕迹"会变成一条与内容无关的线索
```

之后送进 backbone 时中心裁到 **504**（DINOv2 patch=14 的整数倍），同样不 resize。
这些都由 `Inference.py` 自动完成。

---

## 速度

24 GB 显存、batch 8、504×504：约 **1.5 图/秒**（684 ms/图）。

干净图在 L27 停止，省掉 L28~L37 共 13/40 个 block。官方 val 上门把约 60% 的图判为干净，
实测省 **19.7%** 前向深度。

⚠️ 在**原生低分辨率**来源上门会失效——它学到的其实是「图糊不糊」而不是「有没有施加过
退化算子」。实测 200×200 上采样的图里 82% 被判成退化，省下的算力掉到 5.8%。精度不受影响。

---

## 文件

```
vg_shallow_full/stage1.pt   浅组三专家 L14/L21/L27 + 标准化统计量
vg_deep_full/stage1.pt      深组三专家 L26/L33/L37 + 标准化统计量
gate_full.pt                二元门,读浅组特征
*/stage1_history.json       逐 epoch 的 AUC 曲线
```

MIT。DINOv2 权重另有其许可条款。
