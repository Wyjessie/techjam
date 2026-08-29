"""
Attention Residual for AIGC Detection -- the full argument, twelve scenes.

The chain of reasoning, in order:

  00  Proposition          the image is already damaged before it arrives
  01  TwoKnives            different damage kills different evidence
  02  ThreeDetectives      three depths, three kinds of witness
  03  WhyDINOv3            why this backbone and not another
  04  LayerProbe           first measurement: per-layer linear probes
  05  RoutingHypothesis    second measurement -- the hypothesis fails
  06  Committee            three experts, soft routing, AUC(a z) = AUC(z)
  07  EnsembleBeatsAny     the ensemble beats every single layer
  08  WhereToTap           which depths the committee should sit at
  09  TwoBranch            the two-branch policy and its compute model
  10  ParetoResult         a strict Pareto improvement
  11  Closing              what it adds up to

Prose is set in SF Pro and titles in Charter (see architecture.py); formulas
are real LaTeX. Only non-Scene names are imported from architecture.py --
manim discovers scenes by __module__, so nothing from that file leaks into
--write_all here.

Render one scene:
    ./.venv/bin/manim -qh narrative.py RoutingHypothesis
Render all:
    ./.venv/bin/manim -qh narrative.py --write_all

Needs LaTeX on PATH (TinyTeX at ~/Library/TinyTeX/bin/universal-darwin).
"""

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from manim import *

from architecture import (
    T, block, varrow, elbow, annot, Base,
    FONT_TITLE, FONT_BODY, FONT_SMALL,
    FS_HERO, FS_TITLE, FS_LABEL, FS_NOTE,
    BG, C_TEXT, C_DIM, C_FROZEN, C_TRAIN, C_SIGNAL, C_ACC, C_KILL, C_DEAD,
)

# One colour per depth, held from scene 01 to the end.
C_SHALLOW = C_SIGNAL
C_MID     = C_ACC
C_DEEP    = "#b392f0"
C_EXPERT  = (C_SHALLOW, C_MID, C_DEEP)

# LaTeX sets optically smaller than the sans at the same nominal size, so the
# scale in architecture.py gets a matching factor rather than its own numbers.
TEX_RATIO = 1.5


# ---------------------------------------------------------------- helpers

def MT(tex, fs=FS_LABEL, color=C_TEXT, **kw):
    """A formula, on the same type scale as the prose around it."""
    return MathTex(tex, font_size=fs * TEX_RATIO, color=color, **kw)


def rule(width, color=C_DIM, opacity=0.4, sw=1.0):
    return Line(ORIGIN, RIGHT * width, color=color, stroke_width=sw).set_opacity(opacity)


def table(headers, rows, col_w, fs=FS_NOTE, mark=(), dim=(), rh=0.46):
    """Column-aligned table: grey head, one hairline, rows below.

    mark/dim are {(row, col)} sets, so a reading can be pointed at in place
    instead of restating the table underneath it.
    """
    xs, x = [], 0.0
    for w in col_w:
        xs.append(x + w / 2)
        x += w

    def cell(txt, cx, y, w, color, size):
        t = T(txt, size, color)
        if t.width > w - 0.28:
            t.scale((w - 0.28) / t.width)
        return t.move_to([cx, y, 0])

    g = VGroup()
    for h, cx, w in zip(headers, xs, col_w):
        g.add(cell(h, cx, 0.0, w, C_DIM, fs))
    g.add(rule(x).move_to([x / 2, -rh * 0.5, 0]))
    for r, row in enumerate(rows):
        y = -rh * (r + 1) - 0.08
        for c, (txt, cx, w) in enumerate(zip(row, xs, col_w)):
            col = C_ACC if (r, c) in mark else (C_DEAD if (r, c) in dim else C_TEXT)
            g.add(cell(txt, cx, y, w, col, fs))
    return g.move_to(ORIGIN)


def caption(txt, target, buff=0.45, color=C_DIM):
    """Every table of numbers names the set it came from."""
    return T(txt, FS_NOTE, color).next_to(target, DOWN, buff=buff)


def lines(specs, buff=0.28):
    """A stack of body lines, each (text, colour) or (text, colour, size)."""
    g = VGroup()
    for spec in specs:
        g.add(T(spec[0], spec[2] if len(spec) > 2 else FS_LABEL, spec[1]))
    return g.arrange(DOWN, buff=buff)


# ------------------------------------------------- procedural test image
# Scene 00 applies the real operations from its own table -- a real JPEG round
# trip, a real Gaussian blur -- to a synthetic frame. Nothing here claims to be
# a particular sample; what has to be visible is the damage.

def _synth(n=256, seed=7):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    img = np.empty((n, n, 3))
    img[..., 0] = 0.22 + 0.58 * (1 - y)
    img[..., 1] = 0.32 + 0.48 * (1 - y)
    img[..., 2] = 0.52 + 0.38 * (1 - y)
    for cx, cy, r, col in [(0.30, 0.64, 0.17, (0.93, 0.73, 0.34)),
                           (0.70, 0.44, 0.11, (0.86, 0.40, 0.34)),
                           (0.52, 0.82, 0.09, (0.36, 0.72, 0.62))]:
        img[((x - cx) ** 2 + (y - cy) ** 2) < r ** 2] = col
    # fine high-frequency texture -- the first thing any of this destroys
    img += 0.05 * np.sin(60 * np.pi * x)[..., None] * rng.standard_normal((n, n, 1))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _jpeg(a, q):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=q)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def _blur(a, s):
    return np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(s)))


def _crop(a, frac=0.8):
    n = a.shape[0]
    m = int(n * frac)
    o = (n - m) // 2
    return np.array(Image.fromarray(a[o:o + m, o:o + m]).resize((n, n), Image.BICUBIC))


def _tile(arr, label, sub, h=2.3, color=C_DIM):
    """Image plus its name. ImageMobject is not a VMobject, so callers must
    keep these in a Group -- a VGroup will reject it."""
    im = ImageMobject(arr).set_height(h)
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    frame = SurroundingRectangle(im, color=color, stroke_width=1.6, buff=0.0)
    cap = VGroup(T(label, FS_NOTE, C_TEXT), T(sub, FS_NOTE, C_DIM))
    cap.arrange(DOWN, buff=0.08).next_to(im, DOWN, buff=0.22)
    return Group(im, frame, cap)


# =====================================================================
# 00. The proposition
# =====================================================================
class Proposition(Base):
    def construct(self):
        self.chapter("00", "It Is Already Damaged",
                     "the detector never sees the original")

        clean = _synth()
        j = _jpeg(clean, 30)
        jb = _blur(j, 2.0)
        chain = [(j, "JPEG", "q = 30", C_DIM),
                 (jb, "+ blur", "σ = 2.0", C_DIM),
                 (_crop(jb, 0.8), "+ crop", "80%", C_KILL)]

        first = _tile(clean, "as generated", "what the model made", color=C_SIGNAL)
        tiles = [first]
        arrows = VGroup()
        prev = first
        for arr, name, param, col in chain:
            t = _tile(arr, name, param, color=col)
            t.next_to(prev, RIGHT, buff=0.85).align_to(prev, UP)
            arrows.add(Arrow(prev[0].get_right(), t[0].get_left(), buff=0.16,
                             color=C_DIM, stroke_width=2.0,
                             max_tip_length_to_length_ratio=0.3, tip_length=0.15))
            tiles.append(t)
            prev = t
        row = Group(*tiles)
        row.move_to(ORIGIN).shift(UP * 0.6)
        arrows.move_to(row).shift(UP * 0.28)

        self.play(self.fit(row, pad=1.2), run_time=0.4)
        self.play(FadeIn(first, shift=RIGHT * 0.2), run_time=0.9)
        self.wait(0.5)
        for ar, t in zip(arrows, tiles[1:]):
            self.play(GrowArrow(ar), FadeIn(t, shift=RIGHT * 0.2), run_time=0.8)
        self.wait(1.2)

        say = lines([("An AI-generated image is cropped, compressed and filtered", C_TEXT),
                     ("long before it ever reaches a detector.", C_TEXT)], buff=0.22)
        say.next_to(row, DOWN, buff=0.8)
        self.play(self.fit(Group(row, arrows, say), pad=1.12),
                  FadeIn(say, shift=UP * 0.2), run_time=1.0)
        self.wait(2.0)

        ask = lines([("The question is not whether we can catch it.", C_ACC, FS_TITLE),
                     ("It is whether we can still catch it after the damage.",
                      C_ACC, FS_TITLE)], buff=0.32)
        self.play(FadeOut(row), FadeOut(arrows), FadeOut(say), run_time=0.7)
        self.play(self.fit(ask, pad=1.3), FadeIn(ask, shift=UP * 0.2), run_time=1.1)
        self.wait(2.6)

        # ---- the six transforms, at the parameters actually applied ----
        tbl = table(["Transform", "Parameters", "Where it comes from"],
                    [["JPEG compression", "q = 90 / 70 / 50 / 30", "platform transcoding"],
                     ["Gaussian blur", "σ = 0.5 / 1.0 / 2.0", "out of focus"],
                     ["Rescale", "0.5× / 0.25×, then back up", "thumbnails"],
                     ["Gaussian noise", "σ = 0.02 / 0.05 / 0.10", "low-light sensor"],
                     ["Colour jitter", "brightness / contrast / saturation ±20%",
                      "filter apps"],
                     ["Centre crop", "80%", "avatar framing"]],
                    col_w=[4.0, 5.6, 4.6])
        self.play(FadeOut(ask), run_time=0.5)
        self.play(self.fit(tbl, pad=1.22),
                  LaggedStart(*[FadeIn(m, shift=UP * 0.15) for m in tbl],
                              lag_ratio=0.05, run_time=1.8))
        self.wait(3.0)


# =====================================================================
# 01. Two knives, cutting opposite ways
# =====================================================================
class TwoKnives(Base):
    def construct(self):
        self.chapter("01", "Two Knives, Opposite Directions",
                     "different damage kills different evidence")

        spec = [("Shallow", "High-frequency detail",
                 "generator fingerprints: upsampling traces, noise statistics",
                 C_SHALLOW),
                ("Mid", "Medium scale",
                 "texture organisation that does not add up", C_MID),
                ("Deep", "Semantic level",
                 "the sense that the whole thing is not a real photograph", C_DEEP)]
        bars = VGroup()
        for tag, name, desc, col in spec:
            r = RoundedRectangle(width=9.8, height=1.2, corner_radius=0.1,
                                 stroke_color=col, stroke_width=2,
                                 fill_color=col, fill_opacity=0.10)
            lab = T(tag, FS_NOTE, col).move_to(r.get_left() + RIGHT * 1.05)
            txt = VGroup(T(name, FS_LABEL, C_TEXT), T(desc, FS_NOTE, C_DIM))
            txt.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            txt.move_to(r.get_left() + RIGHT * (2.2 + txt.width / 2))
            bars.add(VGroup(r, lab, txt))
        bars.arrange(DOWN, buff=0.45).move_to(ORIGIN)

        self.play(self.fit(bars, pad=1.25), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.3) for b in bars],
                              lag_ratio=0.35, run_time=2.0))
        self.wait(1.0)
        bars.save_state()

        for knife, victim, survivor, msg in [
            ("JPEG compression", 0, 2,
             "the high-frequency forensic signal is gone -- the semantic mismatch is not"),
            ("Centre crop, 80%", 2, 0,
             "global composition is thrown away -- every local pixel trace survives"),
        ]:
            k = T(knife, FS_LABEL, C_KILL).next_to(bars, UP, buff=0.5)
            self.play(FadeIn(k, shift=DOWN * 0.2))
            cross = Cross(bars[victim][0], stroke_color=C_KILL,
                          stroke_width=5).scale(0.98)
            self.play(bars[victim].animate.fade(0.82), Create(cross),
                      bars[survivor].animate.set_stroke(width=4.0), run_time=0.9)
            m = T(msg, FS_NOTE, C_DIM).next_to(bars, DOWN, buff=0.45)
            self.play(FadeIn(m), run_time=0.7)
            self.wait(1.8)
            self.play(FadeOut(cross), FadeOut(m), FadeOut(k), Restore(bars),
                      run_time=0.7)

        self.play(bars.animate.scale(0.7).to_edge(UP, buff=0.5), run_time=0.8)

        ar = Arrow(LEFT * 0.6, RIGHT * 0.6, color=C_KILL, buff=0, stroke_width=3,
                   max_tip_length_to_length_ratio=0.35)
        pair = VGroup(
            VGroup(ar.copy(), MT(r"\text{compression}\;\Rightarrow\;"
                                 r"\text{kills shallow, spares deep}")
                   ).arrange(RIGHT, buff=0.35),
            VGroup(ar.copy().flip(), MT(r"\text{crop}\;\Rightarrow\;"
                                        r"\text{kills deep, spares shallow}")
                   ).arrange(RIGHT, buff=0.35),
        ).arrange(DOWN, buff=0.42, aligned_edge=LEFT).next_to(bars, DOWN, buff=0.6)
        self.play(FadeIn(pair[0], shift=RIGHT * 0.2))
        self.play(FadeIn(pair[1], shift=LEFT * 0.2))
        self.wait(1.0)

        concl = T("No single evidence list is optimal for every kind of damage.",
                  FS_TITLE, C_KILL).next_to(pair, DOWN, buff=0.65)
        self.play(self.fit(VGroup(bars, pair, concl), pad=1.12),
                  FadeIn(concl), run_time=1.2)
        self.wait(3.0)


# =====================================================================
# 02. Three detectives
# =====================================================================
class ThreeDetectives(Base):
    def construct(self):
        self.chapter("02", "Three Detectives",
                     "one backbone, three kinds of witness")

        spec = [("Shallow detective", r"\text{high-frequency fingerprints}", C_SHALLOW),
                ("Mid detective", r"\text{semantics} \;+\; \text{high frequency}", C_MID),
                ("Deep detective", r"\text{semantics}", C_DEEP)]
        cards = VGroup()
        for name, reads, col in spec:
            body = VGroup(T(name, FS_LABEL, C_TEXT),
                          MT(r"\text{reads:}\;\;" + reads, FS_NOTE, col))
            body.arrange(DOWN, buff=0.24)
            r = RoundedRectangle(width=7.4, height=body.height + 0.8,
                                 corner_radius=0.12, stroke_color=col,
                                 stroke_width=2.0, fill_color=col, fill_opacity=0.10)
            body.move_to(r.get_center())
            cards.add(VGroup(r, body))
        cards.arrange(DOWN, buff=0.5).move_to(ORIGIN)

        self.play(self.fit(cards, pad=1.3), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.25) for c in cards],
                              lag_ratio=0.4, run_time=1.8))
        self.wait(1.6)

        free = T("These three detectives are a free lunch inside DINOv3's depth.",
                 FS_TITLE, C_SIGNAL).next_to(cards, DOWN, buff=0.8)
        self.play(self.fit(VGroup(cards, free), pad=1.15),
                  FadeIn(free, shift=UP * 0.2), run_time=1.2)
        self.wait(3.0)


# =====================================================================
# 03. Why this backbone
# =====================================================================
class WhyDINOv3(Base):
    def construct(self):
        self.chapter("03", "Why DINOv3", "three arguments, in order of weight")

        args = [
            ("Self-supervised, so there is no text to lean on",
             "the only supervision available is the image's own internal regularity",
             None, C_FROZEN),
            ("Four of the six transforms are ones it was trained to ignore",
             "blur, rescale, noise and colour jitter are augmentations in its own recipe",
             None, C_SIGNAL),
            ("The token grid lines up exactly",
             "",
             r"\frac{512}{16} = 32 \;\Rightarrow\; 32 \times 32 = 1024"
             r"\;\text{ patch tokens}", C_ACC),
        ]
        cards = VGroup()
        for head, sub, tex, col in args:
            parts = [T(head, FS_LABEL, C_TEXT)]
            if sub:
                parts.append(T(sub, FS_NOTE, C_DIM))
            if tex:
                parts.append(MT(tex, FS_LABEL, col))
            body = VGroup(*parts).arrange(DOWN, buff=0.22)
            r = RoundedRectangle(width=max(11.0, body.width + 1.0),
                                 height=body.height + 0.85, corner_radius=0.12,
                                 stroke_color=col, stroke_width=2.0,
                                 fill_color=col, fill_opacity=0.08)
            body.move_to(r.get_center())
            cards.add(VGroup(r, body))
        cards.arrange(DOWN, buff=0.45).move_to(ORIGIN)

        self.play(self.fit(cards, pad=1.2), run_time=0.4)
        for c in cards:
            self.play(FadeIn(c, shift=UP * 0.25), run_time=0.8)
            self.wait(1.4)
        self.wait(2.0)


# =====================================================================
# 04. First measurement: per-layer linear probes
# =====================================================================
class LayerProbe(Base):
    def construct(self):
        self.chapter("04", "The First Measurement",
                     "one forward pass, thirty-three linear probes")

        proto = lines([
            ("Protocol", C_ACC),
            ("one forward pass, cache the pooled representation at every layer",
             C_DIM, FS_NOTE),
            ("then fit one logistic regression per layer", C_DIM, FS_NOTE),
        ], buff=0.2)
        f1 = MT(r"\hat{y}^{(\ell)} = \sigma\!\left(w_\ell^{\top} f_\ell(x) + b_\ell"
                r"\right), \qquad \ell = 0, 1, \dots, 32")
        f2 = MT(r"\mathrm{AUC}(\ell, d) = \mathrm{AUC}\!\left("
                r"\{w_\ell^{\top} f_\ell(x)\}_{x \in \mathcal{D}_d},\; y\right)")
        head = VGroup(proto, f1, f2).arrange(DOWN, buff=0.5)

        self.play(self.fit(head, pad=1.25), run_time=0.4)
        self.play(FadeIn(proto, shift=UP * 0.2), run_time=0.8)
        self.play(Write(f1), run_time=1.4)
        self.wait(1.0)
        self.play(Write(f2), run_time=1.4)
        self.wait(2.0)
        self.play(FadeOut(head), run_time=0.6)

        # ---- schematic AUC-vs-depth curve ----
        ax = Axes(x_range=[0, 32, 4], y_range=[0.80, 1.0, 0.05],
                  x_length=10.0, y_length=4.4,
                  axis_config={"color": C_DIM, "stroke_width": 1.6,
                               "font_size": FS_NOTE * 1.7},
                  x_axis_config={"numbers_to_include": [8, 16, 24, 32],
                                 "decimal_number_config": {"num_decimal_places": 0}},
                  y_axis_config={"numbers_to_include": [0.85, 0.90, 0.95, 1.00],
                                 "decimal_number_config": {"num_decimal_places": 2}},
                  tips=False)
        ax.set_color(C_DIM)
        xlab = MT(r"\text{layer index }\ell", FS_NOTE, C_DIM)
        xlab.next_to(ax.x_axis, DOWN, buff=0.55)
        ylab = T("AUC", FS_NOTE, C_DIM).next_to(ax.y_axis, UP, buff=0.25)

        def shape(l):
            """Schematic only -- the claims below are the plateau and the dip,
            not any particular per-layer value."""
            rise = 0.86 + 0.105 / (1 + np.exp(-(l - 11) / 2.6))
            dip = 0.055 * np.clip((l - 28) / 4.0, 0, 1) ** 1.6
            return rise - dip

        curve = ax.plot(shape, x_range=[0, 32, 0.25], color=C_FROZEN, stroke_width=3)
        # a vertical highlight, not a filled area -- get_area would drop a slab
        # all the way to y = 0 and read as a bar chart
        band = Rectangle(width=ax.c2p(28, 0)[0] - ax.c2p(20, 0)[0],
                         height=ax.y_length, stroke_width=0,
                         fill_color=C_MID, fill_opacity=0.10)
        band.move_to([(ax.c2p(20, 0)[0] + ax.c2p(28, 0)[0]) / 2,
                      ax.c2p(0, 0.90)[1], 0])

        self.play(self.fit(VGroup(ax, xlab, ylab), pad=1.2), run_time=0.5)
        self.play(Create(ax), FadeIn(xlab), FadeIn(ylab), run_time=1.0)
        self.play(Create(curve), run_time=2.0)
        self.wait(0.8)

        # finding A -- the last layer is not the best layer
        dot_last = Dot(ax.c2p(32, shape(32)), color=C_KILL, radius=0.075)
        dot_peak = Dot(ax.c2p(24, shape(24)), color=C_MID, radius=0.075)
        aA = T("Finding A: the last layer is not the best layer",
               FS_LABEL, C_KILL).next_to(ax, UP, buff=0.45)
        fA = MT(r"\mathrm{AUC}(\ell_{\max}) \;<\; \max_{20 \le \ell \le 28}"
                r"\;\mathrm{AUC}(\ell)", FS_LABEL, C_KILL)
        fA.next_to(ax, DOWN, buff=0.9)
        self.play(FadeIn(dot_last), FadeIn(dot_peak), FadeIn(aA), run_time=0.8)
        self.play(self.fit(VGroup(ax, aA, fA), pad=1.14), Write(fA), run_time=1.3)
        self.wait(2.2)

        # finding B -- a plateau, not a peak
        aB = T("Finding B: layers 20 to 28 are a plateau of strong representation",
               FS_LABEL, C_MID)
        aB.move_to(aA)
        self.play(FadeIn(band), Transform(aA, aB), FadeOut(fA), run_time=1.1)
        self.wait(2.4)

        note = caption("schematic: the claims are the plateau and the dip, "
                       "not any single layer's value", ax, buff=0.95)
        self.play(FadeIn(note), self.fit(VGroup(ax, aA, note), pad=1.14), run_time=0.9)
        self.wait(2.5)


# =====================================================================
# 05. Second measurement: the routing hypothesis fails
# =====================================================================
class RoutingHypothesis(Base):
    def construct(self):
        self.chapter("05", "The Hypothesis That Failed",
                     "the measurement that changed the design")

        hyp = VGroup(
            T("Hypothesis", FS_LABEL, C_ACC),
            MT(r"\ell^{*}(d) \;=\; \arg\max_{\ell}\;\mathrm{AUC}(\ell, d)"
               r"\qquad \text{varies with } d"),
            T("if the best depth depends on the damage, "
              "a router could pick it", FS_NOTE, C_DIM),
        ).arrange(DOWN, buff=0.38)
        self.play(self.fit(hyp, pad=1.3), run_time=0.4)
        self.play(FadeIn(hyp[0]), Write(hyp[1]), run_time=1.5)
        self.play(FadeIn(hyp[2]), run_time=0.7)
        self.wait(2.2)
        self.play(FadeOut(hyp), run_time=0.6)

        rows = [["Blur", "0.9623", "0.9943", "0.9606", "Mid"],
                ["Colour", "0.9456", "1.0000", "0.9949", "Mid"],
                ["JPEG", "0.9567", "0.9060", "0.8339", "Shallow"],
                ["Noise", "0.9857", "0.9900", "0.9603", "Mid"],
                ["Brightness", "0.9576", "0.9975", "0.9874", "Mid"],
                ["Spatial", "0.9665", "0.9912", "0.9409", "Mid"],
                ["Contrast", "0.9807", "0.9989", "0.9959", "Mid"],
                ["Geometric", "0.9572", "0.9947", "0.9854", "Mid"]]
        best = {(0, 2), (1, 2), (2, 1), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2)}
        best |= {(r, 4) for r in range(8)}
        tbl = table(["Degradation", "Shallow  L20", "Mid  L24", "Deep  L28", "best"],
                    rows, col_w=[3.4, 2.9, 2.9, 2.9, 2.4], mark=best)
        cap = caption("per-degradation linear probe, one layer at a time", tbl)

        self.play(self.fit(VGroup(tbl, cap), pad=1.18), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.12) for m in tbl],
                              lag_ratio=0.03, run_time=2.0), FadeIn(cap))
        self.wait(2.0)

        r1 = MT(r"\text{Mid wins } 7/8 \text{ of the degradations}", FS_LABEL, C_MID)
        r1.next_to(cap, DOWN, buff=0.6)
        self.play(self.fit(VGroup(tbl, cap, r1), pad=1.12), Write(r1), run_time=1.2)
        self.wait(1.8)

        # the one real exception
        jpeg_box = SurroundingRectangle(
            VGroup(*[m for m in tbl]).copy(), buff=0.0, stroke_opacity=0)
        r2 = MT(r"\Delta_{\text{JPEG}} = 0.9567 - 0.8339 = 12.3\ \text{pp}"
                r"\;\approx\; 12\,\sigma, \qquad \sigma = 0.010", FS_LABEL, C_KILL)
        r2.next_to(r1, DOWN, buff=0.4)
        self.play(self.fit(VGroup(tbl, cap, r1, r2), pad=1.10),
                  Write(r2), run_time=1.4)
        self.wait(2.6)

        # ---- the turn: everywhere else the gap is inside the noise ----
        self.play(FadeOut(VGroup(tbl, cap, r1, r2)), run_time=0.8)
        turn = MT(r"\forall\, d \ne \text{JPEG}: \qquad \left|\mathrm{AUC}"
                  r"(\ell^{*}, d) - \mathrm{AUC}(\ell_{\text{mid}}, d)\right|"
                  r"\;\lesssim\; \sigma", FS_LABEL, C_TEXT)
        self.play(self.fit(turn, pad=1.4), run_time=0.5)
        self.play(Write(turn), run_time=1.8)
        self.wait(2.8)

        kill = lines([
            ("The degradation state does not reliably predict the best depth.",
             C_KILL, FS_TITLE),
            ("So we do not build an explicit router.", C_ACC, FS_TITLE),
        ], buff=0.36)
        kill.next_to(turn, DOWN, buff=0.9)
        self.play(self.fit(VGroup(turn, kill), pad=1.15),
                  FadeIn(kill[0], shift=UP * 0.2), run_time=1.2)
        self.wait(2.0)
        self.play(FadeIn(kill[1], shift=UP * 0.2), run_time=0.9)
        self.wait(3.5)


# =====================================================================
# 06. The committee, and what soft routing actually is
# =====================================================================
class Committee(Base):
    def construct(self):
        self.chapter("06", "A Committee of Three",
                     "no gate, no router -- just an average")

        heads = MT(r"z_s = h_s\!\left(f_{20}(x)\right), \quad "
                   r"z_m = h_m\!\left(f_{24}(x)\right), \quad "
                   r"z_d = h_d\!\left(f_{28}(x)\right)")
        fuse = MT(r"z \;=\; \tfrac{1}{3}\left(z_s + z_m + z_d\right),"
                  r"\qquad P(\text{AI}) = \sigma(z) = \frac{1}{1 + e^{-z}}")
        top = VGroup(heads, fuse).arrange(DOWN, buff=0.55)
        self.play(self.fit(top, pad=1.3), run_time=0.4)
        self.play(Write(heads), run_time=1.6)
        self.wait(0.8)
        self.play(Write(fuse), run_time=1.6)
        self.wait(1.8)
        self.play(FadeOut(top), run_time=0.6)

        # ---- three placards, one of which will lose its evidence ----
        vals = [2.10, 1.80, 1.50]
        names = ["shallow", "mid", "deep"]
        syms = ["z_s", "z_m", "z_d"]
        cards = VGroup()
        for name, sym, v, col in zip(names, syms, vals, C_EXPERT):
            body = VGroup(T(name + " expert", FS_LABEL, C_TEXT),
                          MT(f"{sym} = {v:+.2f}", FS_LABEL, col))
            body.arrange(DOWN, buff=0.28)
            r = RoundedRectangle(width=3.6, height=body.height + 0.9,
                                 corner_radius=0.12, stroke_color=col,
                                 stroke_width=2.2, fill_color=col, fill_opacity=0.11)
            body.move_to(r.get_center())
            cards.add(VGroup(r, body))
        cards.arrange(RIGHT, buff=0.5)

        tally = MT(r"z = \tfrac{1}{3}(2.10 + 1.80 + 1.50) = 1.80"
                   r"\quad\Rightarrow\quad \sigma(z) = 0.86", FS_LABEL, C_TEXT)
        tally.next_to(cards, DOWN, buff=0.8)
        board = VGroup(cards, tally)

        self.play(self.fit(board, pad=1.2), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.25) for c in cards],
                              lag_ratio=0.3, run_time=1.4))
        self.play(Write(tally), run_time=1.2)
        self.wait(1.8)

        # ---- the damage arrives; one placard goes transparent ----
        hit = T("JPEG at q = 30 -- the shallow expert's evidence is gone",
                FS_LABEL, C_KILL).next_to(cards, UP, buff=0.6)
        self.play(self.fit(VGroup(board, hit), pad=1.14),
                  FadeIn(hit, shift=DOWN * 0.2), run_time=0.9)

        # the whole placard recedes, label included -- a half-faded card with a
        # full-strength name still reads as if it were voting
        dead = VGroup(T("shallow expert", FS_LABEL, C_TEXT),
                      MT(r"z_s \approx 0", FS_LABEL, C_SHALLOW)
                      ).arrange(DOWN, buff=0.28).move_to(cards[0][1])
        dead.set_opacity(0.4)
        self.play(Transform(cards[0][1], dead),
                  cards[0][0].animate.set_stroke(opacity=0.3).set_fill(opacity=0.03),
                  run_time=1.2)
        self.wait(1.0)

        tally2 = MT(r"z \approx \tfrac{1}{3}(0 + 1.80 + 1.50) = 1.10"
                    r"\quad\Rightarrow\quad \sigma(z) = 0.75", FS_LABEL, C_TEXT)
        tally2.move_to(tally)
        self.play(Transform(tally, tally2), run_time=1.2)
        self.wait(1.6)

        same = T("lower confidence -- but the ranking is untouched",
                 FS_NOTE, C_DIM).next_to(tally, DOWN, buff=0.45)
        self.play(FadeIn(same), self.fit(VGroup(board, hit, same), pad=1.12),
                  run_time=0.9)
        self.wait(2.0)

        # ---- why the ranking survives ----
        self.play(FadeOut(VGroup(board, hit, same)), run_time=0.7)
        why = VGroup(
            T("AUC depends only on the ordering of scores,", FS_LABEL, C_TEXT),
            T("and a positive rescaling does not reorder anything:", FS_LABEL, C_TEXT),
            MT(r"\mathrm{AUC}\!\left(\tfrac{1}{3}(z_m + z_d)\right)"
               r"\;=\; \mathrm{AUC}\!\left(z_m + z_d\right)", FS_LABEL, C_SIGNAL),
            MT(r"\mathrm{AUC}(\alpha z) \;=\; \mathrm{AUC}(z)"
               r"\qquad \text{for any } \alpha > 0", FS_LABEL, C_SIGNAL),
        ).arrange(DOWN, buff=0.45)
        self.play(self.fit(why, pad=1.3), run_time=0.5)
        self.play(FadeIn(why[0]), FadeIn(why[1]), run_time=1.0)
        self.wait(1.2)
        self.play(Write(why[2]), run_time=1.4)
        self.wait(1.4)
        self.play(Write(why[3]), run_time=1.4)
        self.wait(2.2)

        pay = T("A failed expert abstains on its own. No gate required.",
                FS_TITLE, C_ACC).next_to(why, DOWN, buff=0.85)
        self.play(self.fit(VGroup(why, pay), pad=1.14),
                  FadeIn(pay, shift=UP * 0.2), run_time=1.2)
        self.wait(3.5)


# =====================================================================
# 07. The ensemble beats every single layer
# =====================================================================
class EnsembleBeatsAny(Base):
    def construct(self):
        self.chapter("07", "Better Than Any One Of Them",
                     "the ensemble is not an average of its members")

        singles = VGroup(
            MT(r"\mathrm{AUC}_{\text{shallow}} = 0.8981", FS_LABEL, C_SHALLOW),
            MT(r"\mathrm{AUC}_{\text{mid}} = 0.9568", FS_LABEL, C_MID),
            MT(r"\mathrm{AUC}_{\text{deep}} = 0.9210", FS_LABEL, C_DEEP),
        ).arrange(RIGHT, buff=0.9)
        ens = MT(r"\mathrm{AUC}_{\text{ens}} = 0.9721", FS_TITLE, C_ACC)
        delta = MT(r"\Delta = \mathrm{AUC}_{\text{ens}} - \mathrm{AUC}_{\text{mid}}"
                   r" = +1.53\ \text{pp}, \qquad \mathrm{SE} = \pm 0.0017"
                   r"\;\Rightarrow\; \approx 9\,\sigma", FS_LABEL, C_TEXT)
        stack = VGroup(singles, ens, delta).arrange(DOWN, buff=0.7)
        cap = caption("NTIRE 10K validation set, robust split", stack, buff=0.55)

        self.play(self.fit(VGroup(stack, cap), pad=1.2), run_time=0.5)
        self.play(LaggedStart(*[Write(s) for s in singles],
                              lag_ratio=0.35, run_time=1.8))
        self.wait(1.2)
        self.play(Write(ens), run_time=1.2)
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(1.4)
        self.play(Write(delta), run_time=1.5)
        self.wait(2.2)

        self.play(FadeOut(VGroup(stack, cap)), run_time=0.7)
        comp = VGroup(
            MT(r"\mathrm{AUC}_{\text{shallow}},\; \mathrm{AUC}_{\text{deep}}"
               r"\;<\; \mathrm{AUC}_{\text{mid}}"
               r"\qquad \text{but} \qquad"
               r"\mathrm{AUC}_{\text{ens}} \;>\; \mathrm{AUC}_{\text{mid}}",
               FS_LABEL, C_TEXT),
            T("Two weaker experts still push the whole thing up.",
              FS_TITLE, C_SIGNAL),
            T("That cannot happen if the three depths carry the same information.",
              FS_LABEL, C_DIM),
        ).arrange(DOWN, buff=0.6)
        self.play(self.fit(comp, pad=1.22), run_time=0.5)
        self.play(Write(comp[0]), run_time=1.6)
        self.wait(1.4)
        self.play(FadeIn(comp[1], shift=UP * 0.2), run_time=0.9)
        self.wait(1.2)
        self.play(FadeIn(comp[2]), run_time=0.8)
        self.wait(3.2)


# =====================================================================
# 08. Where the committee should sit
# =====================================================================
class WhereToTap(Base):
    def construct(self):
        self.chapter("08", "Where To Tap",
                     "clean images and damaged images want opposite depths")

        tbl = table(["Tap layers", "Layers to run", "clean", "robust"],
                    [["12 / 16 / 20", "20", "0.9887", "0.9583"],
                     ["16 / 20 / 24", "24", "0.9881", "0.9698"],
                     ["20 / 24 / 28", "28", "0.9850", "0.9721"],
                     ["24 / 28 / 32", "32", "0.9788", "0.9681"]],
                    col_w=[3.6, 3.2, 2.8, 2.8],
                    mark={(0, 2), (2, 3)})
        cap = caption("NTIRE 10K validation set -- same protocol as scene 07", tbl)

        self.play(self.fit(VGroup(tbl, cap), pad=1.25), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.12) for m in tbl],
                              lag_ratio=0.04, run_time=1.6), FadeIn(cap))
        self.wait(2.4)

        trend = MT(r"\frac{\partial\, \mathrm{AUC}_{\text{clean}}}{\partial \ell}"
                   r"\;<\; 0, \qquad \mathrm{AUC}_{\text{robust}}(\ell)"
                   r"\;\text{ rises, then falls}", FS_LABEL, C_TEXT)
        trend.next_to(cap, DOWN, buff=0.7)
        self.play(self.fit(VGroup(tbl, cap, trend), pad=1.14),
                  Write(trend), run_time=1.5)
        self.wait(2.0)

        why = lines([
            ("On a clean image the low-level fingerprint is intact,", C_DIM),
            ("and a shallow layer can read it directly.", C_DIM),
            ("Once damage wipes it out, you have to go deeper", C_ACC),
            ("for the semantic evidence that survived.", C_ACC),
        ], buff=0.24)
        why.next_to(trend, DOWN, buff=0.7)
        self.play(self.fit(VGroup(tbl, cap, trend, why), pad=1.10),
                  FadeIn(why, shift=UP * 0.2), run_time=1.2)
        self.wait(3.5)


# =====================================================================
# 09. The two-branch policy and its compute model
# =====================================================================
class TwoBranch(Base):
    def construct(self):
        self.chapter("09", "Two Branches, One Dial",
                     "what the threshold buys and what it costs")

        syms = MT(r"\begin{aligned}"
                  r"L_s,\, L_d \;&:\; \text{layers the shallow / deep branch runs"
                  r"\;(20 / 28)}\\[2pt]"
                  r"p \;&:\; \text{share of genuinely clean images in the traffic}"
                  r"\\[2pt]"
                  r"\sigma_c(\tau) \;&:\; \text{clean images judged clean}"
                  r"\;\rightarrow\; \text{shallow branch}\\[2pt]"
                  r"\sigma_d(\tau) \;&:\; \text{damaged images misjudged clean}"
                  r"\;\rightarrow\; \text{shallow branch}\\[2pt]"
                  r"c_{\det} \;&:\; \text{cost of the discriminator itself,"
                  r"\; in layers}"
                  r"\end{aligned}", FS_NOTE)
        self.play(self.fit(syms, pad=1.3), run_time=0.4)
        self.play(FadeIn(syms, shift=UP * 0.2), run_time=1.2)
        self.wait(3.0)
        self.play(FadeOut(syms), run_time=0.6)

        f1 = MT(r"f(p, \tau) \;=\; p\,\sigma_c(\tau) + (1 - p)\,\sigma_d(\tau)")
        c1 = T("the share that takes the shallow branch", FS_NOTE, C_DIM)
        f2 = MT(r"\bar{L}(p, \tau) \;=\; c_{\det} + L_d - (L_d - L_s)"
                r"\Big[\, p\,\sigma_c(\tau) + (1 - p)\,\sigma_d(\tau) \,\Big]")
        c2 = T("expected compute", FS_NOTE, C_DIM)
        f3 = MT(r"\frac{\bar{L}}{L_d} \;=\; \frac{c_{\det}}{L_d} + 1"
                r" - \frac{L_d - L_s}{L_d}\Big[\, p\,\sigma_c"
                r" + (1 - p)\,\sigma_d \,\Big]")
        c3 = T("relative to running everything deep", FS_NOTE, C_DIM)

        model = VGroup(*[VGroup(c, f).arrange(DOWN, buff=0.2)
                         for c, f in [(c1, f1), (c2, f2), (c3, f3)]])
        model.arrange(DOWN, buff=0.6)
        self.play(self.fit(model, pad=1.22), run_time=0.5)
        for grp in model:
            self.play(FadeIn(grp[0]), Write(grp[1]), run_time=1.4)
            self.wait(1.0)
        self.wait(1.4)

        ceil = MT(r"\frac{L_d - L_s}{L_d} \;=\; \frac{8}{28} \;=\; 28.6\%",
                  FS_TITLE, C_ACC)
        ceil_c = T("the ceiling: what you would save if everything went shallow",
                   FS_NOTE, C_DIM)
        cg = VGroup(ceil, ceil_c).arrange(DOWN, buff=0.35)
        self.play(FadeOut(model), run_time=0.6)
        self.play(self.fit(cg, pad=1.4), Write(ceil), run_time=1.4)
        self.play(FadeIn(ceil_c), run_time=0.6)
        self.wait(2.4)
        self.play(FadeOut(cg), run_time=0.6)

        # ---- the property that makes the dial safe to turn ----
        key = VGroup(
            T("Accuracy does not depend on the traffic mix", FS_LABEL, C_TEXT),
            MT(r"A_{\text{robust}} = A_r(\sigma_d), \qquad "
               r"A_{\text{clean}} = A_c(\sigma_c)", FS_LABEL, C_SIGNAL),
        ).arrange(DOWN, buff=0.45)
        self.play(self.fit(key, pad=1.35), run_time=0.5)
        self.play(FadeIn(key[0]), run_time=0.7)
        self.play(Write(key[1]), run_time=1.4)
        self.wait(1.8)

        pay = lines([
            ("p enters the cost, never the accuracy.", C_ACC, FS_TITLE),
            ("So set the threshold from the accuracy you can tolerate,", C_DIM),
            ("then compute the cost from your own traffic.", C_DIM),
        ], buff=0.3).next_to(key, DOWN, buff=0.8)
        self.play(self.fit(VGroup(key, pay), pad=1.15),
                  FadeIn(pay, shift=UP * 0.2), run_time=1.2)
        self.wait(3.5)


# =====================================================================
# 10. The result: a strict Pareto improvement
# =====================================================================
class ParetoResult(Base):
    def construct(self):
        self.chapter("10", "A Strict Pareto Improvement",
                     "cheaper, and not worse anywhere")

        sweep = table(["threshold", "shallow share", "clean", "robust",
                       "mean layers"],
                      [["−∞", "0.00", "0.9850", "0.9721", "28.0"],
                       ["−11.65", "0.10", "0.9860", "0.9715", "27.2"],
                       ["−8.71", "0.20", "0.9864", "0.9710", "26.4"],
                       ["+∞", "1.00", "0.9887", "0.9583", "20.0"]],
                      col_w=[2.6, 3.0, 2.6, 2.6, 2.8],
                      mark={(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)})
        head = T("Threshold sweep", FS_LABEL, C_ACC).next_to(sweep, UP, buff=0.5)
        cap = caption("same evaluation as scene 08 -- the tap-depth table", sweep)

        self.play(self.fit(VGroup(head, sweep, cap), pad=1.2), run_time=0.5)
        self.play(FadeIn(head),
                  LaggedStart(*[FadeIn(m, shift=UP * 0.12) for m in sweep],
                              lag_ratio=0.04, run_time=1.6), FadeIn(cap))
        self.wait(3.0)
        self.play(FadeOut(VGroup(head, sweep, cap)), run_time=0.7)

        final = table(["", "all deep", "binary routing", "change"],
                      [["Clean AUC", "0.9536", "0.9584", "+0.48 pp"],
                       ["Robust AUC", "0.8634", "0.8646", "+0.12 pp"],
                       ["Mean layers", "28.00", "26.41", "−5.7%"]],
                      col_w=[3.2, 2.8, 3.4, 2.8],
                      mark={(0, 3), (2, 3)})
        head2 = T("At a traffic mix of 30% clean / 70% damaged",
                  FS_LABEL, C_ACC).next_to(final, UP, buff=0.5)
        # Production note: these numbers come from a different configuration
        # than scene 08's, so the caption says so on screen rather than
        # leaving the audience to reconcile 0.9850 with 0.9536.
        cap2 = caption("a different evaluation configuration from scene 08 -- "
                       "not directly comparable with the numbers there", final)
        se = MT(r"\mathrm{SE} = \pm 0.35\ \text{pp}", FS_NOTE, C_DIM)
        se.next_to(cap2, DOWN, buff=0.3)

        self.play(self.fit(VGroup(head2, final, cap2, se), pad=1.18), run_time=0.5)
        self.play(FadeIn(head2),
                  LaggedStart(*[FadeIn(m, shift=UP * 0.12) for m in final],
                              lag_ratio=0.05, run_time=1.6))
        self.play(FadeIn(cap2), Write(se), run_time=1.0)
        self.wait(2.0)

        noise = T("robust is inside the noise band -- it did not get worse",
                  FS_NOTE, C_DIM).next_to(se, DOWN, buff=0.35)
        self.play(FadeIn(noise),
                  self.fit(VGroup(head2, final, cap2, se, noise), pad=1.12),
                  run_time=0.9)
        self.wait(2.4)

        self.play(FadeOut(VGroup(head2, final, cap2, se, noise)), run_time=0.7)
        pareto = MT(r"\Delta A_{\text{robust}} \approx 0 \;\;\wedge\;\;"
                    r"\Delta A_{\text{clean}} > 0 \;\;\wedge\;\;"
                    r"\Delta \bar{L} < 0", FS_LABEL, C_TEXT)
        verdict = T("A strict Pareto improvement.", FS_TITLE, C_ACC)
        pg = VGroup(pareto, verdict).arrange(DOWN, buff=0.8)
        self.play(self.fit(pg, pad=1.35), run_time=0.5)
        self.play(Write(pareto), run_time=1.8)
        self.wait(1.4)
        self.play(FadeIn(verdict, shift=UP * 0.2), run_time=1.0)
        self.wait(3.5)


# =====================================================================
# 11. What it adds up to
# =====================================================================
class Closing(Base):
    def construct(self):
        self.chapter("11", "What It Adds Up To", "")

        sums = MT(r"\underbrace{\text{three detectives}}_{\text{three depth taps}}"
                  r"\;\;+\;\;"
                  r"\underbrace{\text{one forward pass}}_{\text{frozen backbone}}"
                  r"\;\;+\;\;"
                  r"\underbrace{\text{zero extra cost}}"
                  r"_{\text{the intermediate layers are a by-product}}",
                  FS_LABEL, C_TEXT)
        self.play(self.fit(sums, pad=1.35), run_time=0.5)
        self.play(Write(sums), run_time=2.6)
        self.wait(3.0)

        foot = MT(r"\text{NTIRE: } 24{,}000 \text{ train} \;/\; "
                  r"80{,}000 \text{ test}, \qquad"
                  r"\texttt{val\_hard}: 5{,}000 \;"
                  r"(50\%\ \text{clean} \;/\; 50\%\ \text{damaged})",
                  FS_NOTE, C_DIM)
        foot.next_to(sums, DOWN, buff=1.1)
        self.play(self.fit(VGroup(sums, foot), pad=1.2),
                  FadeIn(foot), run_time=1.2)
        self.wait(4.0)
