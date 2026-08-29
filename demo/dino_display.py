from manim import *

# ============ 改这里就够了 ============
N_LAYERS = 33
FOCUS_RANGE = (20, 28)        # 拉近后聚焦的层段
EXPERT_LAYERS = [20, 24, 28]   # 三个专家取的层
EXPERT_NAMES = ["Shallow Detective", "Mid Detective", "Deep Detective"]
EXPERT_DESCS = ["High-Freq Fingerprints", "Texture & Structure", "Semantic Manifold"]
EXPERT_COLORS = [TEAL, YELLOW, PURPLE]
CJK_FONT = "PingFang SC"      # macOS 自带；换系统改这里
# =====================================


def CText(t, **kw):
    """带中文字体的 Text"""
    return Text(t, font=CJK_FONT, **kw)


class DinoExperts(MovingCameraScene):   # 用到 camera.frame，必须是 MovingCameraScene
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ---------- 1. 画 DINOv3 全架构 ----------
        title = CText("DINOv3", font_size=36).to_edge(UP)

        layers = VGroup(*[
            Rectangle(width=2.2, height=0.22,
                      stroke_width=1.2,
                      stroke_color=GREY_B,
                      fill_color=GREY_D, fill_opacity=0.5)
            for _ in range(N_LAYERS)
        ]).arrange(DOWN, buff=0.045)
        layers.set_height(5.8).next_to(title, DOWN, buff=0.25)

        self.play(FadeIn(title), LaggedStart(
            *[FadeIn(l, shift=RIGHT * 0.2) for l in layers],
            lag_ratio=0.03, run_time=2))
        self.wait(0.5)

        # ---------- 2. 高亮聚焦区间 ----------
        lo, hi = FOCUS_RANGE
        focus = VGroup(*layers[lo:hi + 1])
        rest = VGroup(*[l for i, l in enumerate(layers)
                        if not lo <= i <= hi])          # 不要和 focus 重叠
        box = SurroundingRectangle(focus, color=YELLOW, buff=0.06,
                                   stroke_width=2)
        label = CText(f"layer {lo}-{hi}", font_size=22, color=YELLOW)
        label.next_to(box, RIGHT, buff=0.3)

        self.play(Create(box), FadeIn(label))
        self.play(rest.animate.set_opacity(0.25),
                  focus.animate.set_opacity(1.0))
        self.wait(0.5)

        # ---------- 3. 镜头拉近 ----------
        self.play(
            self.camera.frame.animate.scale(0.45).move_to(focus),
            FadeOut(title), FadeOut(label),
            run_time=1.8)
        self.wait(0.3)

        # ---------- 4. 三个专家拉出来 ----------
        self.play(self.camera.frame.animate.scale(1 / 0.45).move_to(ORIGIN),
                  FadeOut(box), run_time=1.2)

        picked = [layers[i] for i in EXPERT_LAYERS]
        cards = VGroup()
        for lyr, name, desc, col in zip(picked, EXPERT_NAMES,
                                        EXPERT_DESCS, EXPERT_COLORS):
            lyr.set_color(col).set_opacity(1.0)
            card = VGroup(
                RoundedRectangle(width=3.4, height=1.5, corner_radius=0.12,
                                 stroke_color=col, fill_color=col,
                                 fill_opacity=0.12),
                CText(name, font_size=22, color=col),
                CText(desc, font_size=17, color=GREY_A),
            )
            card[1].move_to(card[0].get_center() + UP * 0.25)
            card[2].move_to(card[0].get_center() + DOWN * 0.28)
            cards.add(card)

        cards.arrange(DOWN, buff=0.45).to_edge(RIGHT, buff=1.0)

        # 先把架构挪到左边，再算箭头端点（顺序反了箭头会指错位置）
        self.play(layers.animate.to_edge(LEFT, buff=1.2), run_time=1.0)

        arrows = VGroup(*[
            Arrow(lyr.get_right(), card.get_left(), buff=0.12,
                  stroke_width=2.5, color=col,
                  max_tip_length_to_length_ratio=0.12)
            for lyr, card, col in zip(picked, cards, EXPERT_COLORS)
        ])

        for card, arrow in zip(cards, arrows):
            self.play(GrowArrow(arrow), FadeIn(card, shift=LEFT * 0.3),
                      run_time=0.7)
        self.wait(1)

        # ---------- 5. 收尾 ----------
        self.wait(2)
