"""SQuaDE 推理 —— 门控两组专家,干净图在 L27 提前退出。

    python Inference.py --dir /path/to/images \
        --shallow runs/mix_shallow --deep runs/mix_deep --gate runs/mix_gate.pt \
        --out preds.csv

    # 单图
    python Inference.py --image a.jpg --shallow ... --deep ... --gate ...

输出 CSV: image, score(0~1 越大越可能是假), label(0真/1假), route(shallow/deep), depth_used

---------------------------------------------------------------------------
两组专家 + 门

    输入 → DINOv2 ViT-g(冻结)
             ├─ 跑到 L27,取 L14/L21/L27 → 浅组三专家均匀平均 → z_shallow
             │                          → 门 g(读同样的浅层特征)
             │        g 判"干净" → 直接输出 z_shallow,**在这里停止**,省掉 L28~L37
             └─ g 判"退化" → 继续跑到 L37,取 L26/L33/L37 → 深组 → z_deep

门读的是**浅组的特征**,这一点是架构上的关键:它必须在提前退出点之前就能算出来。
门若改读深层特征,就必须先跑完全深度,提前退出的收益整个消失。

实测(28 万模型,官方 val):省 19.7% 前向深度,robust AUC 不降反升。
但在**原生低分辨率**来源上门会失效(把 82% 的图判成退化,省下的算力掉到 5.8%),
因为它学到的实际是"图糊不糊"而不是"有没有施加过退化算子"。

---------------------------------------------------------------------------
预处理必须与训练完全一致,否则等于给测试集加一层没记录的域偏移

    短边 >= 512   **中心裁 512x512,一次重采样都不做**
                  —— 保住原生高频(生成器指纹住在那里),也消除"缩放倍率"这条捷径
    短边 <  512   先裁成正方,再用**随机挑的一个内核**上采样到 512
                  内核池 [BILINEAR, BICUBIC, BOX, LANCZOS],按图确定性挑
                  —— 固定一个内核的话,"这张图带哪种插值痕迹"会变成一条与内容无关的线索

然后喂给 backbone 时 **crop 到 504**(DINOv2 patch=14 的整数倍),同样不 resize。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.dinov3 import DINOv3Backbone                        # noqa: E402
from models.experts_mlp import ExpertBank                       # noqa: E402
from utils.preprocess import INTERP_POOL, normalize, rng_for    # noqa: E402

SHALLOW_LAYERS = [14, 21, 27]
DEEP_LAYERS = [26, 33, 37]
EXIT_DEPTH, FULL_DEPTH = 27, 37        # 提前退出点 / 完整深度(总 40 个 block)


def pick_kernel(name: str):
    """按图名确定性地挑一个上采样内核 —— 与训练时 --interp random 同一套逻辑。"""
    r = rng_for(0, name + "|interp")
    return INTERP_POOL[int(r.integers(len(INTERP_POOL)))]


def load_image(p: Path, size: int = 512) -> Image.Image:
    """短边 >=size 中心裁不重采样;<size 裁正方后随机内核上采样。"""
    with Image.open(p) as im:
        im = im.convert("RGB")
        kernel = pick_kernel(p.name) if min(im.size) < size else None
        return normalize(im, size, kernel, fit="crop")


class SQuaDE:
    """门控两组专家 + 真正的提前退出。"""

    def __init__(self, shallow_run, deep_run, gate_path, model, crop=504, device="cuda"):
        self.dev, self.crop = device, crop
        self.bb = DINOv3Backbone(name=model, device=device)
        self.shallow = self._bank(shallow_run)
        self.deep = self._bank(deep_run)
        g = torch.load(gate_path, map_location="cpu")
        self.gate = nn.Sequential(nn.Linear(g["D"], g["hidden"]), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(g["hidden"], 1))
        self.gate.load_state_dict(g["net"])
        self.gate.to(device).eval()
        self.g_mu, self.g_sd = g["mu"].to(device), g["sd"].to(device)
        self.prep = self.bb.make_preprocessor(crop_size=crop)

    def _bank(self, run):
        ck = torch.load(Path(run) / "stage1.pt", map_location="cpu")
        b = ExpertBank(hidden_size=ck["cfg"]["hidden_size"], hidden=ck["cfg"]["hidden"],
                       dropout=ck["cfg"]["dropout"])
        b.load_state_dict(ck["bank"])
        b.to(self.dev).freeze_experts()
        return b

    @staticmethod
    def _pack(feats, layers, dev):
        """{layer: entry} -> (B,3,3H) 与 (B,3,2),顺序按 layers 给定。"""
        f = torch.stack([DINOv3Backbone.pool(feats[l]) for l in layers], 1).to(dev, torch.float32)
        p = torch.stack([feats[l]["prenorm_stats"] for l in layers], 1).to(dev, torch.float32)
        return f, p

    @torch.no_grad()
    def predict(self, imgs, names):
        # 预处理器每张返回 (1,3,H,W),要 cat 不是 stack —— stack 会多出一维
        x = torch.cat([self.prep(im, image_id=n) for im, n in zip(imgs, names)])

        # ① 只跑到 L27。门与浅组都读这一段的特征,所以判"干净"时后面 13 个 block 根本不跑。
        f1, hid = self.bb.forward_blocks(x, layers=SHALLOW_LAYERS + [DEEP_LAYERS[0]],
                                         max_block=EXIT_DEPTH)
        fs, ps = self._pack(f1, SHALLOW_LAYERS, self.dev)
        zs, ps_parts = self.shallow(fs, ps, return_parts=True)
        votes = ps_parts["expert_logits"].clone()          # (B,3) 组内三个专家各自的 logit
        gz = self.gate((fs.reshape(len(imgs), -1) - self.g_mu) / self.g_sd).squeeze(-1)
        clean = gz <= 0

        z = zs.clone()
        n_deep = int((~clean).sum())
        if n_deep:
            # ② 只有被判为退化的图才继续跑 L28~L37,而且从 L27 的 hidden 接着跑,不重跑前面
            sel = (~clean).nonzero(as_tuple=True)[0]
            f2, _ = self.bb.forward_blocks(None, layers=DEEP_LAYERS[1:], max_block=FULL_DEPTH,
                                           resume=hid[sel], start_block=EXIT_DEPTH)
            f2[DEEP_LAYERS[0]] = {k: (v[sel] if torch.is_tensor(v) else v)
                                  for k, v in f1[DEEP_LAYERS[0]].items()}
            fd, pd = self._pack(f2, DEEP_LAYERS, self.dev)
            zd, pd_parts = self.deep(fd, pd, return_parts=True)
            z[sel] = zd
            votes[sel] = pd_parts["expert_logits"]
        return (torch.sigmoid(z).cpu().numpy(), clean.cpu().numpy(),
                gz.cpu().numpy(), torch.sigmoid(votes).cpu().numpy())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", help="图片目录(递归)")
    src.add_argument("--image", help="单张图")
    ap.add_argument("--shallow", required=True, help="浅组 run 目录(含 stage1.pt)")
    ap.add_argument("--deep", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--model", default="facebook/dinov2-giant")
    ap.add_argument("--out", default=None, help="输出 CSV;不给则打印")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--crop-size", type=int, default=504)
    ap.add_argument("--no-early-exit", action="store_true",
                    help="关掉提前退出:两组都算完再按门选。精度相同,只是不省算力,"
                         "用来核对提前退出没有改变结果")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    paths = ([Path(a.image)] if a.image else
             sorted(p for p in Path(a.dir).rglob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")))
    if not paths:
        raise SystemExit("没有找到图片")
    print(f"图片 {len(paths)} 张   浅组 {SHALLOW_LAYERS}   深组 {DEEP_LAYERS}", flush=True)

    net = SQuaDE(a.shallow, a.deep, a.gate, a.model, a.crop_size, a.device)

    import time
    rows, n_exit, t_total = [], 0, 0.0
    for k in range(0, len(paths), a.batch_size):
        chunk = paths[k : k + a.batch_size]
        imgs = [load_image(p) for p in chunk]
        t0 = time.perf_counter()
        prob, cl, gz, votes = net.predict(imgs, [p.name for p in chunk])
        dt = time.perf_counter() - t0
        t_total += dt
        n_exit += int(cl.sum())
        for q, pr, c, g, v in zip(chunk, prob, cl, gz, votes):
            grp = SHALLOW_LAYERS if c else DEEP_LAYERS
            rows.append({"image": str(q), "score": f"{pr:.6f}", "label": int(pr > 0.5),
                         "route": "shallow" if c else "deep",
                         "experts": "/".join(f"L{l}" for l in grp),
                         # 组内三个专家各自的概率 —— 三票分歧大说明这张图处在决策边界上
                         "votes": "|".join(f"{x:.3f}" for x in v),
                         "vote_spread": f"{float(v.max() - v.min()):.3f}",
                         "depth_used": EXIT_DEPTH if c else FULL_DEPTH,
                         "gate_logit": f"{g:.4f}",
                         "ms_per_img": f"{dt / len(chunk) * 1000:.1f}"})
        if (k // a.batch_size) % 20 == 0:
            print(f"  {min(k + a.batch_size, len(paths))}/{len(paths)}", flush=True)

    import statistics as _st
    saved = n_exit / len(paths) * (1 - EXIT_DEPTH / 40)
    print(f"\n判为干净(走浅组,可提前退出): {n_exit}/{len(paths)} = {n_exit/len(paths)*100:.1f}%")
    print(f"提前退出省下的前向深度: {saved*100:.1f}%  "
          f"(干净图跑到 L{EXIT_DEPTH} 就停,省掉 L{EXIT_DEPTH+1}~L{FULL_DEPTH} 共 13/40 个 block)")
    print(f"判为假: {sum(r['label'] for r in rows)}/{len(rows)}")
    print(f"\n用时: 共 {t_total:.2f} s   平均 {t_total / len(paths) * 1000:.1f} ms/图   "
          f"吞吐 {len(paths) / t_total:.1f} 图/秒")
    sp = [float(r["vote_spread"]) for r in rows]
    print(f"组内三专家分歧(max-min 概率): 中位 {_st.median(sp):.3f}  最大 {max(sp):.3f}")
    print("  分歧大的图处在决策边界上 —— 三个专家看同一张图给出不同答案,这类样本最值得人工复核")

    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"-> {a.out}")
    else:
        for r in rows[:20]:
            print(f"  {Path(r['image']).name:<34} {r['score']}  "
                  f"{'假' if r['label'] else '真'}  {r['route']:<8} {r['experts']:<14} "
                  f"投票 {r['votes']}")
        if len(rows) > 20:
            print(f"  ... 共 {len(rows)} 行,用 --out 写 CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
