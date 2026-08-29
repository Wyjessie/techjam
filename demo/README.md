# Manim scenes — AIGC detection talk

Two Manim scripts used as visual aids for the attention-residual AIGC
detection talk. Not the webapp demo (see the root [README.md](../README.md)
for that) — these render to video.

- `architecture.py` — five scenes walking through the design motivation and
  architecture, "why → how".
- `narrative.py` — the full argument as twelve scenes, imports shared
  helpers from `architecture.py` (module must stay alongside it). Needs
  LaTeX on `PATH` (TinyTeX at `~/Library/TinyTeX/bin/universal-darwin`) —
  formulas are rendered as real LaTeX, not text.
- `dino_display.py` — a DINOv3-layer visualization (uses Chinese-language
  labels/comments; edit the config block at the top to reconfigure).

## Setup

Install the system deps first — `pip install` below builds `pycairo` from
source and fails without `pkg-config`/`cairo` already on the machine:

```bash
# macOS
brew install pkg-config cairo pango ffmpeg

# Debian/Ubuntu
apt install pkg-config libcairo2-dev libpango1.0-dev ffmpeg
```

Then, Python 3.13 (developed on 3.13.5):

```bash
cd demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Both scripts render text with system fonts rather than bundling any:
`architecture.py` uses Charter, SF Pro Display, and SF Pro Text;
`dino_display.py` uses PingFang SC (macOS-only — swap `CJK_FONT` at the top
of the file if rendering elsewhere). Missing fonts won't error, they'll just
fall back silently to a default typeface.

## Render

`-ql` = quick/low quality preview, `-qh` = high quality, `-p` = open the
result when done.

`architecture.py` — scenes, in the "why → how" order the file's docstring
lays out:

```bash
manim -ql architecture.py WhyAdaptive       # physical intuition: corruptions kill different evidence
manim -ql architecture.py ParadigmSplit     # invariance (flatten) vs adaptation (exploit)
manim -ql architecture.py EstimatorDetail   # degradation estimator CNN
manim -ql architecture.py RouterDetail      # weights MLP router
manim -ql architecture.py FullArchitecture  # everything put together
```

High quality, opening the result:

```bash
manim -pqh architecture.py FullArchitecture
```

All scenes in the file at once:

```bash
manim -ql architecture.py --write_all
```

`dino_display.py` — one scene:

```bash
manim -pqh dino_display.py DinoExperts
```

`narrative.py` — twelve scenes, the chain of reasoning in order
(`Proposition`, `TwoKnives`, `ThreeDetectives`, `WhyDINOv3`, `LayerProbe`,
`RoutingHypothesis`, `Committee`, `EnsembleBeatsAny`, `WhereToTap`,
`TwoBranch`, `ParetoResult`, `Closing`):

```bash
manim -qh narrative.py RoutingHypothesis
manim -qh narrative.py --write_all
```

## Output

Rendered video lands in `media/videos/<script name>/<quality>/<SceneName>.mp4`
(e.g. `media/videos/architecture/480p15/FullArchitecture.mp4`), created next
to wherever you ran `manim` from. That directory is gitignored — don't check
rendered output in.
