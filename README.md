# Keepout

**A fixed camera over machinery, watched by something that never blinks.**

Entry for the OpenCV AI Competition 2026.

Live endpoint: <https://mqrmfp6mbi.eu-west-2.awsapprunner.com>

> "In an effort to raise the alarm, he repeatedly waved at a CCTV camera in the
> hope that someone monitoring the system would spot him and come to his aid,
> nobody did."
>
> — HSE, 9 September 2026, on Factory Services UK Limited of Knowsley, after a
> worker's arm was pulled into an unguarded conveyor.

A camera was already pointed at him. The footage existed. Nobody was watching it.

---

## Where things are

| Path | What it is |
|---|---|
| **[`products/keepout/`](products/keepout/)** | The product. Start with its [README](products/keepout/README.md). |
| [`products/keepout/docs/report.md`](products/keepout/docs/report.md) | The technical report the competition rules require. |
| [`products/keepout/docs/evaluation.md`](products/keepout/docs/evaluation.md) | Every measured number, and how to reproduce it. |
| [`products/keepout/docs/architecture.md`](products/keepout/docs/architecture.md) | Pipeline and AWS diagrams. |
| [`products/keepout/docs/costs.md`](products/keepout/docs/costs.md) | What was created on AWS and what it costs. |
| `packages/visioncore` | Shared OpenCV 5 primitives: IO, models, records, timing, calibration. |
| `packages/servicekit` | Shared FastAPI service shell: jobs, SSE progress, evidence routes. |
| `infra/` | ECR, App Runner and S3 scripts. |
| `research/` | The verified source material. `EVIDENCE.md` carries the OSHA and HSE citations with dates and quotes; `FINDINGS.md` carries the OpenCV 5 and AWS facts. |

## Quick start

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python \
  -e packages/visioncore -e packages/servicekit -e products/keepout

keepout build-samples          # stage the labelled evaluation clips
keepout demo --bake            # pre-compute what the service serves
uvicorn keepout.service:app --port 8000
```

```bash
python -m pytest products/keepout/tests     # 121 tests
keepout evaluate                            # reproduce every figure in the docs
```

Requires `ffmpeg` on the path for `build-samples`, and pins
`opencv-python-headless==5.0.0.93`. That pin matters: OpenCV 4.14.0 was released
after 5.0.0, so an unpinned install resolves to 4.x.

## Results

| | |
|---|---|
| Detection rate inside the zone | 819 / 819 frames, 95% CI [0.9955, 1.0] |
| Time to alert | 0 ms, median and maximum, 4 of 4 labelled entries |
| False alerts, empty room | 0 in 0.0111 camera-hours |
| Incidents raised while the view was unusable | 0 across 228 blind frames |
| Throughput | 26 ms/frame on 22 threads, 440 ms/frame on AWS App Runner |

And the one that argues against us: on 40 seconds of real crowded pedestrian
footage the person-unaccounted rule produced one false critical, about 90 per
camera-hour in that setting. It produced none on the staged machine-cell clips.

**No deployed-system outcome trial exists for this class of product.** If you ask
whether this has been shown to prevent an injury in a real factory, the honest
answer is that nobody has published that evidence, for any product in this
category.

## Licence

This is a competition entry and is not published under an open-source licence.
Third-party components keep their own: OpenCV Apache-2.0, YOLOX Apache-2.0,
`vtest.avi` Apache-2.0, NumPy BSD-3, FastAPI MIT. Nothing here is AGPL.
