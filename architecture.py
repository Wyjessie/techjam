"""
Attention Residual for AIGC Detection -- design motivation and details.
Five scenes, ordered "why -> how":

  1  WhyAdaptive       Physical intuition: different corruptions kill different evidence
  2  ParadigmSplit     Paradigm split: invariance (flatten) vs adaptation (exploit)
  3  EstimatorDetail   Degradation estimator CNN -- the free "coroner"
  4  RouterDetail      Weights MLP -- a tiny, auditable switch
  5  FullArchitecture  Putting it together

Render one scene:
    ./.venv/bin/manim -ql architecture.py WhyAdaptive
Render all:
    ./.venv/bin/manim -ql architecture.py --write_all
"""

from manim import *

# ============ Global style ============
# Three families, one job each.
FONT_TITLE   = "Charter"          # serif -- chapter titles and payoff lines
FONT_BODY    = "SF Pro Display"   # sans -- labels and statements
FONT_SMALL   = "SF Pro Text"      # sans, small optical size -- notes, captions

OPTICAL_CUT  = 20            # Apple's own Display/Text crossover
RENDER_FS    = 64            # glyphs are laid out at this size, then scaled

# Type scale -- four steps, each a clear jump. The serif takes over at
# FS_TITLE, so size and family change together and nothing needs to go bold.
FS_HERO  = 90   # chapter numeral -- decorative
FS_TITLE = 42   # chapter title, and the one payoff line per scene
FS_LABEL = 24   # module names, body statements
FS_NOTE  = 16   # subtitles, annotations, captions, legend

BG       = "#0d1117"
C_TEXT   = "#e6e1dc"
C_DIM    = "#7d8590"
C_FROZEN = "#589bd5"   # frozen module
C_TRAIN  = "#e8836b"   # trainable module
C_SIGNAL = "#4db6a0"   # degradation conditioning signal
C_ACC    = "#f0c674"   # emphasis / annotation
C_KILL   = "#e5534b"   # killed evidence
C_DEAD   = "#484f58"
# =================================


def _font_for(fs):
    if fs >= FS_TITLE:
        return FONT_TITLE
    return FONT_BODY if fs >= OPTICAL_CUT else FONT_SMALL


def T(t, fs=FS_LABEL, color=C_TEXT, weight=NORMAL, font=None, **kw):
    """Serif above FS_TITLE, sans below; contrast comes from the family switch
    rather than the weight axis, so NORMAL is the default everywhere.

    Laid out at RENDER_FS and scaled down -- manim's glyph positioning rounds
    badly below ~20, which shreds SF Pro Text's tracking at subtitle sizes.
    """
    return Text(t, font=font or _font_for(fs), font_size=RENDER_FS, color=color,
                weight=weight, **kw).scale(fs / RENDER_FS)


def block(title, sub="", w=4.0, h=1.05, color=C_TRAIN, fill=0.13,
          fs=FS_LABEL, sfs=FS_NOTE, radius=0.12, dashed=False):
    """Rounded block: title + grey subtitle.

    The box grows to the type, never the other way round -- shrinking text to
    fit a box is what flattens a type scale back into mush.
    """
    parts = [T(title, fs, C_TEXT)]
    if sub:
        parts.append(T(sub, sfs, C_DIM, line_spacing=0.65))
    g = VGroup(*parts).arrange(DOWN, buff=0.14)
    w = max(w, g.width + 0.56)
    h = max(h, g.height + 0.40)
    r = RoundedRectangle(width=w, height=h, corner_radius=radius,
                         stroke_color=color, stroke_width=2.0,
                         fill_color=color, fill_opacity=fill)
    if dashed:
        r = DashedVMobject(r, num_dashes=48, equal_lengths=False)
        r.set_stroke(color, 1.8)
    g.move_to(r.get_center())
    return VGroup(r, g)


def varrow(a, b, color=C_DIM, sw=2.0):
    return Arrow(a.get_bottom(), b.get_top(), buff=0.1, stroke_width=sw,
                 color=color, max_tip_length_to_length_ratio=0.25, tip_length=0.16)


def elbow(a, b, color=C_DIM, sw=1.8):
    """Right-angle connector, so arrows never cut through a block."""
    st, en = a.get_bottom(), b.get_top()
    mid = (st[1] + en[1]) / 2
    return VGroup(
        Line(st, [st[0], mid, 0], color=color, stroke_width=sw),
        Line([st[0], mid, 0], [en[0], mid, 0], color=color, stroke_width=sw),
        Arrow([en[0], mid, 0], en, color=color, buff=0, stroke_width=sw,
              max_tip_length_to_length_ratio=0.4, tip_length=0.14),
    )


def annot(text, target, direction=RIGHT, color=C_ACC, fs=FS_NOTE, buff=0.5):
    """Dashed side annotation next to a block."""
    t = T(text, fs, color, line_spacing=0.65).next_to(target, direction, buff=buff)
    ln = DashedLine(target.get_edge_center(direction),
                    t.get_edge_center(-direction),
                    dash_length=0.07, stroke_width=1.2, color=color)
    ln.set_opacity(0.55)
    return VGroup(ln, t)


class Base(MovingCameraScene):
    def setup(self):
        super().setup()
        self.camera.background_color = BG

    def fit(self, mob, pad=1.16):
        """Camera move that actually contains `mob` -- fixed widths clip type
        the moment the scale changes."""
        f = self.camera.frame
        ar = f.width / f.height
        w = max(mob.width * pad, mob.height * pad * ar)
        return f.animate.set_width(w).move_to(mob)

    def chapter(self, num, title, sub=""):
        n = T(num, FS_HERO, C_DIM, weight=THIN, font=FONT_BODY).set_opacity(0.35)
        t = T(title, FS_TITLE, C_TEXT)
        s = T(sub, FS_LABEL, C_ACC) if sub else None
        grp = VGroup(t, *( [s] if s else [] )).arrange(DOWN, buff=0.3)
        n.next_to(grp, UP, buff=0.35)
        all_ = VGroup(n, grp).move_to(ORIGIN)
        self.play(FadeIn(n, shift=DOWN * 0.2), FadeIn(t, shift=UP * 0.2), run_time=0.9)
        if s:
            self.play(FadeIn(s), run_time=0.5)
        self.wait(1.2)
        self.play(FadeOut(all_), run_time=0.6)


# =====================================================================
# 1. Physical intuition: different corruptions kill different evidence
# =====================================================================
class WhyAdaptive(Base):
    def construct(self):
        self.chapter("01", "Corruption Kills Evidence",
                     "but different corruptions kill different evidence")

        head = T("In an AI-generated image, evidence lives in different places",
                 FS_LABEL).to_edge(UP, buff=0.6)
        self.play(FadeIn(head))

        rows = [
            ("Shallow", "High-frequency fingerprints",
             "Generator upsampling traces live in fine detail", C_SIGNAL),
            ("Mid", "Texture & structure artifacts",
             "Unnatural local statistics live at mid scale", C_ACC),
            ("Deep", "Semantic manifold drift",
             "\"Doesn't look like a real photo\" lives in semantics", "#b392f0"),
        ]
        bars = VGroup()
        for tag, name, desc, col in rows:
            r = RoundedRectangle(width=8.4, height=1.0, corner_radius=0.1,
                                 stroke_color=col, stroke_width=2,
                                 fill_color=col, fill_opacity=0.10)
            lab = T(tag, FS_NOTE, col).move_to(r.get_left() + RIGHT * 0.95)
            nm = T(name, FS_LABEL, C_TEXT)
            ds = T(desc, FS_NOTE, C_DIM)
            txt = VGroup(nm, ds).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            txt.move_to(r.get_left() + RIGHT * (2.05 + txt.width / 2))
            bars.add(VGroup(r, lab, txt))
        bars.arrange(DOWN, buff=0.42).next_to(head, DOWN, buff=0.7)

        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.3) for b in bars],
                              lag_ratio=0.35, run_time=2.2))
        self.wait(1.0)

        bars.save_state()

        note = T("Real-world corruption is a selective killer", FS_LABEL, C_KILL)
        note.next_to(bars, DOWN, buff=0.55)
        self.play(FadeIn(note))
        self.wait(0.8)

        # ---- First knife: JPEG ----
        knife = T("JPEG compression - hard quantization", FS_LABEL, C_KILL).move_to(note)
        self.play(Transform(note, knife))
        cross = Cross(bars[0][0], stroke_color=C_KILL, stroke_width=5).scale(0.98)
        self.play(bars[0].animate.fade(0.80), Create(cross), run_time=0.9)
        msg1 = VGroup(
            T("The high-frequency forensic signal is erased", FS_NOTE, C_DIM),
            T("but the semantic inconsistency survives -- compression", FS_NOTE, C_DIM),
            T("will not fix a shadow pointing the wrong way", FS_NOTE, C_DIM),
        ).arrange(DOWN, buff=0.12).next_to(note, DOWN, buff=0.35)
        self.play(bars[2].animate.set_stroke(width=4.0).scale(1.025),
                  FadeIn(msg1), run_time=0.9)
        self.wait(1.8)

        self.play(FadeOut(cross), FadeOut(msg1), Restore(bars), run_time=0.7)

        # ---- Second knife: crop ----
        knife2 = T("Center crop 80% - content discarded", FS_LABEL, C_KILL).move_to(note)
        self.play(Transform(note, knife2))
        cross2 = Cross(bars[2][0], stroke_color=C_KILL, stroke_width=5).scale(0.98)
        self.play(bars[2].animate.fade(0.80), Create(cross2), run_time=0.9)
        msg2 = VGroup(
            T("Global composition cues are thrown away", FS_NOTE, C_DIM),
            T("but every local pixel-level trace is untouched", FS_NOTE, C_DIM),
        ).arrange(DOWN, buff=0.12).next_to(note, DOWN, buff=0.35)
        self.play(bars[0].animate.set_stroke(width=4.0).scale(1.025),
                  FadeIn(msg2), run_time=0.9)
        self.wait(1.8)

        # ---- Conclusion: the two knives cut opposite ways ----
        self.play(FadeOut(cross2), FadeOut(msg2), FadeOut(note), FadeOut(head),
                  Restore(bars), run_time=0.7)
        self.play(bars.animate.scale(0.72).to_edge(UP, buff=0.45), run_time=0.8)

        a1 = T("Compression: kills shallow evidence, spares deep", FS_LABEL, C_TEXT)
        a2 = T("Cropping: kills deep evidence, spares shallow", FS_LABEL, C_TEXT)
        arrow_r = Arrow(LEFT * 0.6, RIGHT * 0.6, color=C_KILL, buff=0,
                        stroke_width=3, max_tip_length_to_length_ratio=0.35)
        arrow_l = Arrow(RIGHT * 0.6, LEFT * 0.6, color=C_KILL, buff=0,
                        stroke_width=3, max_tip_length_to_length_ratio=0.35)
        r1 = VGroup(arrow_r.copy(), a1).arrange(RIGHT, buff=0.35)
        r2 = VGroup(arrow_l.copy(), a2).arrange(RIGHT, buff=0.35)
        pair = VGroup(r1, r2).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        pair.next_to(bars, DOWN, buff=0.5)
        self.play(FadeIn(r1, shift=RIGHT * 0.2))
        self.play(FadeIn(r2, shift=LEFT * 0.2))
        self.wait(0.8)

        concl = VGroup(
            T("The two knives cut in opposite directions.", FS_TITLE, C_KILL),
            T("So no single evidence list is optimal for every corruption --", FS_LABEL, C_TEXT),
            T("which evidence survives depends on what the image went through.",
              FS_LABEL, C_ACC),
        ).arrange(DOWN, buff=0.24).next_to(pair, DOWN, buff=0.5)
        self.play(FadeIn(concl[0]))
        self.wait(0.5)
        self.play(FadeIn(concl[1]), FadeIn(concl[2]))
        self.wait(2.5)


# =====================================================================
# 2. Paradigm split: invariance vs adaptation
# =====================================================================
class ParadigmSplit(Base):
    def construct(self):
        self.chapter("02", "Adapt, Don't Flatten", "adaptation vs invariance")

        div = DashedLine(UP * 3.4, DOWN * 3.4, color=C_DIM,
                         dash_length=0.12, stroke_width=1.2).set_opacity(0.4)
        lt = T("Existing paradigm - Invariance", FS_LABEL, C_DIM).move_to(LEFT * 3.5 + UP * 3.0)
        rt = T("Ours - Adaptation", FS_LABEL, C_SIGNAL).move_to(RIGHT * 3.5 + UP * 3.0)
        ls = T("LPT / DCPT / GlobalForge", FS_NOTE, C_DIM).next_to(lt, DOWN, buff=0.15)
        self.play(Create(div), FadeIn(lt), FadeIn(rt), FadeIn(ls))

        def cloud(center, n, color, spread=0.55, seed_shift=0.0):
            g = VGroup()
            for i in range(n):
                a = (i * 2.39996 + seed_shift)
                rad = spread * ((i + 1) / n) ** 0.5
                d = Dot(center + np.array([rad * np.cos(a), rad * np.sin(a), 0]),
                        radius=0.055, color=color)
                g.add(d)
            return g

        # ---- Left: clean and degraded forced together ----
        lc = LEFT * 3.5 + UP * 0.9
        clean_l = cloud(lc + LEFT * 1.1, 9, C_FROZEN)
        degr_l = cloud(lc + RIGHT * 1.1, 9, C_KILL, seed_shift=1.1)
        cap_l = VGroup(T("clean", FS_NOTE, C_FROZEN).next_to(clean_l, DOWN, buff=0.3),
                       T("degraded", FS_NOTE, C_KILL).next_to(degr_l, DOWN, buff=0.3))
        self.play(FadeIn(clean_l), FadeIn(degr_l), FadeIn(cap_l))
        self.wait(0.5)

        pull = T("Forced onto the same point in feature space", FS_NOTE, C_DIM).next_to(
            cap_l, DOWN, buff=0.5)
        self.play(FadeIn(pull))
        self.play(clean_l.animate.move_to(lc), degr_l.animate.move_to(lc),
                  FadeOut(cap_l), run_time=1.4)
        cost = VGroup(T("The cost: the model is asked to stay confident", FS_NOTE, C_KILL),
                      T("about information that no longer exists", FS_NOTE, C_KILL)
                      ).arrange(DOWN, buff=0.13).next_to(pull, DOWN, buff=0.45)
        self.play(FadeIn(cost))
        self.wait(1.5)

        # ---- Right: keep the difference, use it as a routing signal ----
        rc = RIGHT * 3.5 + UP * 0.9
        clean_r = cloud(rc + LEFT * 1.1 + UP * 0.2, 9, C_FROZEN)
        degr_r = cloud(rc + RIGHT * 1.1 + DOWN * 0.2, 9, C_KILL, seed_shift=1.1)
        self.play(FadeIn(clean_r), FadeIn(degr_r))
        keep = T("The difference is left alone", FS_NOTE, C_DIM).move_to(
            RIGHT * 3.5 + pull.get_y() * UP)
        self.play(FadeIn(keep))

        sig = CurvedArrow(degr_r.get_bottom() + DOWN * 0.15,
                          rc + DOWN * 1.9, angle=-0.9, color=C_SIGNAL,
                          stroke_width=2.5, tip_length=0.18)
        gain = VGroup(T("Degradation state is not noise to remove --", FS_NOTE, C_SIGNAL),
                      T("it is a free routing signal", FS_NOTE, C_SIGNAL)
                      ).arrange(DOWN, buff=0.13)
        gain.move_to(RIGHT * 3.5 + cost.get_y() * UP)
        self.play(Create(sig), FadeIn(gain))
        self.wait(2.0)

        # ---- Bottom: one corroborating fact ----
        self.play(FadeOut(VGroup(clean_l, degr_l, clean_r, degr_r, sig,
                                 pull, keep, div, lt, rt, ls)),
                  VGroup(cost, gain).animate.shift(UP * 2.2), run_time=1.0)
        fact = VGroup(
            T("NTIRE 2026 - a competition with 511 registrants", FS_LABEL, C_DIM),
            T("20 teams all pushed invariance: bigger models, more data,", FS_LABEL, C_TEXT),
            T("harder augmentation, more ensembling", FS_LABEL, C_TEXT),
            T("Not one team made the input adaptive.", FS_TITLE, C_ACC),
        ).arrange(DOWN, buff=0.28).shift(DOWN * 1.3)
        self.play(LaggedStart(*[FadeIn(f, shift=UP * 0.2) for f in fact],
                              lag_ratio=0.5, run_time=2.0))
        self.wait(2.5)


# =====================================================================
# 3. Degradation estimator CNN
# =====================================================================
class EstimatorDetail(Base):
    def construct(self):
        self.chapter("03", "Autopsy First, Then Pick the Eye",
                     "Degradation estimator - frozen - labels cost nothing")

        blocks = [
            block("Raw crop 256x256", "no resize - no normalization",
                  w=5.0, color=C_DIM, fill=0.06),
            block("stem - stride = 1 throughout",
                  "4 x conv3x3  ->  receptive field 9x9",
                  w=5.8, color=C_TRAIN),
            block("GroupNorm + SiLU", "", w=4.0, h=0.7, color=C_DIM, fill=0.06, fs=FS_LABEL),
            block("stages - stride = 2 each", "32 -> 64 -> 96 -> 128 -> 160",
                  w=5.8, color=C_TRAIN),
            block("mean + std dual pooling", "", w=4.8, h=0.75, color=C_DIM,
                  fill=0.06, fs=FS_LABEL),
            block("384-dim feature h", "", w=4.0, h=0.7, color=C_ACC,
                  fill=0.10, fs=FS_LABEL),
        ]
        flow = VGroup(*blocks).arrange(DOWN, buff=0.55)
        flow.move_to(ORIGIN)

        self.camera.frame.set_height(5.2).move_to(blocks[0])

        notes = {
            1: "9x9 exactly covers JPEG's 8-pixel block period\n"
               "stride=1: never discard the high frequencies we came for",
            3: "GroupNorm, not BatchNorm\n"
               "degradation strength must not be diluted by batch statistics",
            4: "Noise is a variance quantity\nmean pooling alone cannot see it",
        }

        prev = None
        for i, b in enumerate(blocks):
            anims = [FadeIn(b, shift=UP * 0.25)]
            if prev is not None:
                anims.append(GrowArrow(varrow(prev, b)))
            anims.append(self.camera.frame.animate.move_to(b))
            self.play(*anims, run_time=0.85)
            if i in notes:
                a = annot(notes[i], b, RIGHT, buff=0.55)
                grp = VGroup(b, a)
                self.play(Create(a[0]), FadeIn(a[1]),
                          self.fit(grp, pad=1.3), run_time=0.75)
                self.wait(1.5)
                self.play(FadeOut(a),
                          self.camera.frame.animate.set_height(5.2).move_to(b),
                          run_time=0.5)
            prev = b

        # Scale bypass
        bypass = VGroup(
            Line(blocks[1].get_right(), blocks[1].get_right() + RIGHT * 1.5,
                 color=C_SIGNAL, stroke_width=2),
            Line(blocks[1].get_right() + RIGHT * 1.5,
                 blocks[4].get_right() + RIGHT * 1.5, color=C_SIGNAL, stroke_width=2),
            Arrow(blocks[4].get_right() + RIGHT * 1.5, blocks[4].get_right(),
                  color=C_SIGNAL, buff=0, stroke_width=2,
                  max_tip_length_to_length_ratio=0.3, tip_length=0.15),
        )
        blab = T("scale bypass\n(pre-normalization)", FS_NOTE, C_SIGNAL, line_spacing=0.65)
        blab.next_to(bypass[1], RIGHT, buff=0.2)
        span = VGroup(blocks[1], blocks[4], bypass, blab)
        self.play(self.fit(span, pad=1.25), run_time=0.8)
        self.play(Create(bypass), FadeIn(blab), run_time=1.0)
        self.wait(1.6)

        # ---- Six heads + CORAL ----
        heads_spec = [("jpeg", "5 lv"), ("blur", "4 lv"), ("noise", "4 lv"),
                      ("resize", "3 lv"), ("jitter", "2 lv"), ("crop", "3 lv")]
        heads = VGroup(*[
            block(n, k, w=1.55, h=0.85, color=C_TRAIN, fs=FS_LABEL, sfs=FS_NOTE, radius=0.09)
            for n, k in heads_spec
        ]).arrange(RIGHT, buff=0.18)
        heads.next_to(blocks[-1], DOWN, buff=1.0)

        self.play(self.fit(VGroup(blocks[-1], heads)), run_time=0.9)

        fan = VGroup(*[
            Line(blocks[-1].get_bottom(), h.get_top(), color=C_DIM,
                 stroke_width=1.4).set_opacity(0.7) for h in heads
        ])
        hlab = T("Six independent heads -- attributes co-occur, they are not "
                 "mutually exclusive classes", FS_NOTE, C_DIM)
        hlab.next_to(heads, UP, buff=0.30)
        hlab.add_background_rectangle(color=BG, opacity=1.0, buff=0.09)
        self.play(Create(fan), FadeIn(heads), FadeIn(hlab), run_time=1.2)
        self.wait(1.2)

        coral = block("CORAL ordinal head    logit_k = w . h + b_k",
                      "shared w, monotone b_k  ->  rank consistency by construction",
                      w=7.8, color=C_TRAIN)
        coral.next_to(heads, DOWN, buff=0.9)
        fan2 = VGroup(*[
            Line(h.get_bottom(), coral.get_top(), color=C_DIM,
                 stroke_width=1.2).set_opacity(0.55) for h in heads
        ])
        self.play(self.fit(VGroup(heads, coral)), run_time=0.6)
        self.play(Create(fan2), FadeIn(coral), run_time=1.0)
        a = annot("\"mild\" and \"severe\" are ordered\n"
                  "a plain classifier would treat them as unrelated classes",
                  coral, RIGHT, buff=0.5)
        cg = VGroup(coral, a)
        self.play(Create(a[0]), FadeIn(a[1]),
                  self.fit(cg, pad=1.25), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(a))

        out = block("6-dim discrete code  +  continuous soft code", "",
                    w=7.0, h=0.85, color=C_ACC, fill=0.10, fs=FS_LABEL)
        out.next_to(coral, DOWN, buff=0.7)
        self.play(GrowArrow(varrow(coral, out)), FadeIn(out),
                  self.fit(VGroup(coral, out)), run_time=0.9)
        self.wait(1.0)

        free = T("This step is nearly free: we apply the corruptions ourselves, "
                 "so the labels cost nothing", FS_NOTE, C_SIGNAL
                 ).next_to(out, DOWN, buff=0.6)
        # the six heads have done their job -- clear them so the closing
        # statement isn't framed against a sliver of cropped boxes
        self.play(FadeOut(VGroup(heads, fan, fan2, hlab)), FadeIn(free),
                  self.fit(VGroup(coral, out, free), pad=1.18), run_time=0.8)
        self.wait(2.5)


# =====================================================================
# 4. Weights MLP -- a tiny, auditable switch
# =====================================================================
class RouterDetail(Base):
    def construct(self):
        self.chapter("04", "Intelligence Must Live Where It Can Be Audited",
                     "Weights MLP - about 900 parameters in total")

        wires = VGroup()
        src = block("Degradation code (from the frozen estimator)",
                    "6-dim discrete  +  6-dim soft code",
                    w=6.4, color=C_DIM, fill=0.06)
        src.to_edge(UP, buff=0.9)
        self.play(FadeIn(src))

        left = block("one-hot expand -> 24 dims",
                     "independent parameters per level,\nlearns non-linearity",
                     w=4.6, h=1.25, color=C_TRAIN)
        right = block("soft code -> 6 dims",
                      "order preserving,\ninterpolates unseen strengths",
                      w=4.6, h=1.25, color=C_TRAIN)
        two = VGroup(left, right).arrange(RIGHT, buff=0.7)
        two.next_to(src, DOWN, buff=0.75)
        wires.add(elbow(src, left), elbow(src, right))
        self.play(Create(wires[0]), Create(wires[1]), FadeIn(two), run_time=1.1)
        self.wait(1.2)

        neck = block("30-dim input - information bottleneck",
                     "cannot see any image content",
                     w=6.2, h=1.15, color=C_KILL, fill=0.14, fs=FS_LABEL)
        neck.next_to(two, DOWN, buff=0.8)
        wires.add(elbow(left, neck), elbow(right, neck))
        self.play(Create(wires[2]), Create(wires[3]), FadeIn(neck), run_time=1.0)

        guard = block("Structural guarantee",
                      "routing can only be a function\nof the degradation state",
                      w=4.0, h=1.05, color=C_DIM, fill=0.0, fs=FS_LABEL, sfs=FS_NOTE,
                      dashed=True)
        guard.next_to(neck, RIGHT, buff=0.6)
        link = DashedLine(neck.get_right(), guard.get_left(), color=C_DIM,
                          dash_length=0.06, stroke_width=1.2).set_opacity(0.6)
        self.play(Create(link), FadeIn(guard))
        self.wait(2.0)

        self.play(VGroup(src, two, neck, guard, link, wires
                         ).animate.shift(UP * 1.35).scale(0.82), run_time=0.8)

        l1 = block("Linear 30 -> 32  +  GELU", "", w=4.8, h=0.8, color=C_TRAIN, fs=FS_LABEL)
        l2 = block("Linear 32 -> 3  (zero-init)",
                   "starts exactly uniform  (1/3, 1/3, 1/3)",
                   w=5.6, color=C_TRAIN)
        sm = block("softmax( . / T )", "temperature T:  2.0 -> 1.0, annealed",
                   w=5.0, color=C_DIM, fill=0.06)
        tail = VGroup(l1, l2, sm).arrange(DOWN, buff=0.38)
        tail.next_to(neck, DOWN, buff=0.5)

        tarrows = VGroup()
        prev = neck
        for b in tail:
            ar = varrow(prev, b)
            tarrows.add(ar)
            self.play(GrowArrow(ar), FadeIn(b, shift=UP * 0.2),
                      self.camera.frame.animate.move_to(
                          VGroup(prev, b).get_center()), run_time=0.7)
            prev = b
        self.wait(0.8)

        a = annot("zero-init = no routing preference at the start\n"
                  "routing is learned, not stumbled into by initialization",
                  l2, RIGHT, buff=0.45)
        ag = VGroup(l2, a)
        self.play(Create(a[0]), FadeIn(a[1]),
                  self.fit(ag, pad=1.25), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(a), self.camera.frame.animate.set_width(
            14.22).move_to(l2), run_time=0.5)

        ws = VGroup(*[
            block(t, s, w=2.3, h=0.8, color=c, fill=0.12, fs=FS_LABEL, sfs=FS_NOTE)
            for t, s, c in [("w1", "shallow", C_SIGNAL), ("w2", "mid", C_ACC),
                            ("w3", "deep", "#b392f0")]
        ]).arrange(RIGHT, buff=0.4)
        ws.next_to(sm, DOWN, buff=0.7)
        fan = VGroup(*[Line(sm.get_bottom(), w.get_top(), color=C_DIM,
                            stroke_width=1.4).set_opacity(0.7) for w in ws])
        total = T("about 900 parameters total  -  weighted sum over the three "
                  "experts' logits", FS_NOTE, C_DIM)
        total.next_to(ws, DOWN, buff=0.45)
        lower = VGroup(neck, tail, ws, total)
        self.play(Create(fan), FadeIn(ws),
                  self.fit(lower, pad=1.12), run_time=1.1)
        self.play(FadeIn(total))
        self.wait(1.8)

        self.play(FadeOut(VGroup(src, two, neck, guard, link, wires,
                                 tail, tarrows, ws, fan, total)),
                  self.camera.frame.animate.set_height(8.0).move_to(ORIGIN),
                  run_time=1.0)
        final = VGroup(
            T("Every adaptive degree of freedom collapses into two lookup tables:",
              FS_LABEL, C_TEXT),
            T("one decides which layer to read from  (~100 parameters)", FS_LABEL, C_ACC),
            T("one decides how to calibrate what is read  (~10,000 parameters)",
              FS_LABEL, C_ACC),
            T("Each switch can be ablated on its own; every routing decision "
              "can be printed out.", FS_NOTE, C_SIGNAL),
        ).arrange(DOWN, buff=0.35)
        self.play(LaggedStart(*[FadeIn(f, shift=UP * 0.2) for f in final],
                              lag_ratio=0.45, run_time=2.4))
        self.wait(2.5)


# =====================================================================
# 5. Putting it together
# =====================================================================
class FullArchitecture(Base):
    def construct(self):
        self.chapter("05", "Putting It Together",
                     "a dumb backbone  +  a tiny auditable switch")

        X = block("Input image", "", w=3.0, h=0.7, color=C_DIM, fill=0.06, fs=FS_LABEL)
        X.to_edge(UP, buff=0.55)

        # columns sit on their own measured widths, not magic numbers
        LX, RX = -4.20, 3.17

        est = block("Degradation estimator (small CNN)",
                    "pretrained on free labels - frozen",
                    w=4.4, color=C_FROZEN, fill=0.12)
        code = block("6-dim discrete degradation vector",
                     "jpeg - blur - resize\nnoise - jitter - crop",
                     w=4.4, h=1.25, color=C_DIM, fill=0.06)
        wmlp = block("Weights MLP", "softmax -> (w1, w2, w3)",
                     w=4.4, color=C_TRAIN)
        lcol = VGroup(est, code, wmlp).arrange(DOWN, buff=0.55)
        lcol.next_to(X, DOWN, buff=0.9).set_x(LX)

        dino_bands = VGroup(*[
            block(t, "", w=6.4, h=0.6, color=C_FROZEN, fill=0.07, fs=FS_NOTE, radius=0.07)
            for t in ["shallow band - high-frequency cues",
                      "mid band - texture & structure cues",
                      "deep band - semantic manifold cues"]
        ]).arrange(DOWN, buff=0.16)
        dino_box = RoundedRectangle(width=7.4, height=3.15, corner_radius=0.14,
                                    stroke_color=C_FROZEN, stroke_width=2.2,
                                    fill_color=C_FROZEN, fill_opacity=0.10)
        dino_t = VGroup(T("Frozen DINOv3", FS_LABEL, C_FROZEN),
                        T("three depth bands chosen by E0 probing", FS_NOTE, C_DIM)
                        ).arrange(DOWN, buff=0.08)
        dino_t.move_to(dino_box.get_top() + DOWN * 0.48)
        dino_bands.move_to(dino_box.get_bottom() + UP * 1.12)
        dino = VGroup(dino_box, dino_t, dino_bands)
        dino.next_to(X, DOWN, buff=0.9).set_x(RX)

        self.play(FadeIn(X))
        self.play(FadeIn(dino, shift=UP * 0.2), run_time=0.9)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in lcol],
                              lag_ratio=0.4, run_time=1.5))

        self.play(Create(elbow(X, est)), Create(elbow(X, dino)),
                  GrowArrow(varrow(est, code)), GrowArrow(varrow(code, wmlp)),
                  run_time=1.0)
        self.wait(0.8)

        experts = VGroup(*[
            block(n, s, w=1.72, h=1.0, color=C_TRAIN, fs=FS_LABEL, sfs=FS_NOTE, radius=0.09)
            for n, s in [("shallow expert", "pool+MLP -> z1"),
                         ("mid expert", "pool+MLP -> z2"),
                         ("deep expert", "pool+MLP -> z3")]
        ]).arrange(RIGHT, buff=0.2)
        experts.next_to(dino, DOWN, buff=0.7).set_x(RX)
        taps = VGroup(*[
            Arrow([ex.get_x(), dino_box.get_bottom()[1], 0], ex.get_top(),
                  color=C_DIM, buff=0.05, stroke_width=1.8,
                  max_tip_length_to_length_ratio=0.35, tip_length=0.14)
            for ex in experts])
        self.play(Create(taps), FadeIn(experts), run_time=1.1)

        fuse = block("z  =  w1*z1 + w2*z2 + w3*z3",
                     "weighted fusion in logit space",
                     w=5.0, color=C_DIM, fill=0.06, fs=FS_LABEL)
        fuse.next_to(experts, DOWN, buff=0.7).set_x(RX - 0.3)
        self.play(*[Create(elbow(ex, fuse)) for ex in experts],
                  FadeIn(fuse), run_time=1.0)

        # green conditioning signal
        sy = fuse.get_y()
        cond = VGroup(
            Line(wmlp.get_right(), [wmlp.get_right()[0] + 0.5, wmlp.get_y(), 0],
                 color=C_SIGNAL, stroke_width=2.4),
            Line([wmlp.get_right()[0] + 0.5, wmlp.get_y(), 0],
                 [wmlp.get_right()[0] + 0.5, sy, 0], color=C_SIGNAL, stroke_width=2.4),
            Arrow([wmlp.get_right()[0] + 0.5, sy, 0], fuse.get_left(),
                  color=C_SIGNAL, buff=0.05, stroke_width=2.4,
                  max_tip_length_to_length_ratio=0.25, tip_length=0.16),
        )
        self.play(Create(cond), run_time=1.0)

        out = block("real / fake confidence  =  sigmoid(z)", "", w=5.0, h=0.8,
                    color=C_ACC, fill=0.10, fs=FS_LABEL)
        out.next_to(fuse, DOWN, buff=0.6).set_x(RX - 0.3)
        self.play(GrowArrow(varrow(fuse, out)), FadeIn(out), run_time=0.8)
        self.wait(1.0)

        # legend
        def key(color, label, fill):
            sq = RoundedRectangle(width=0.3, height=0.24, corner_radius=0.05,
                                  stroke_color=color, stroke_width=1.8,
                                  fill_color=color, fill_opacity=fill)
            return VGroup(sq, T(label, FS_NOTE, C_DIM)).arrange(RIGHT, buff=0.15)
        line_key = VGroup(Line(ORIGIN, RIGHT * 0.35, color=C_SIGNAL, stroke_width=2.4),
                          T("degradation conditioning signal (6-dim bottleneck)",
                            FS_NOTE, C_DIM)).arrange(RIGHT, buff=0.15)
        legend = VGroup(key(C_FROZEN, "frozen", 0.12), key(C_TRAIN, "trainable", 0.13),
                        line_key).arrange(RIGHT, buff=0.6)
        legend.next_to(out, DOWN, buff=0.5).set_x(0)

        whole = VGroup(X, lcol, dino, taps, experts, fuse, out, cond, legend)
        self.play(self.fit(whole, pad=1.10), FadeIn(legend), run_time=1.4)
        self.wait(3.0)
