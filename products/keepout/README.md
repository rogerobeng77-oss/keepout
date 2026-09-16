# Keepout

**A fixed camera over machinery, watched by something that never blinks.**

Live: <https://mqrmfp6mbi.eu-west-2.awsapprunner.com>

> "In an effort to raise the alarm, he repeatedly waved at a CCTV camera in the
> hope that someone monitoring the system would spot him and come to his aid,
> nobody did."
>
> — HSE press release, 9 September 2026, on Factory Services UK Limited of
> Knowsley, after a worker's arm was pulled into an unguarded conveyor.
> [Source](https://press.hse.gov.uk/2026/09/09/manufacturer-fined-after-worker-suffers-life-changing-injuries-in-conveyor-incident/)

A camera was already pointed at him. The footage existed. Nobody was watching it.
Keepout is the part that was missing.

---

## What it does

1. **Learns the danger zone.** An operator draws it once against a reference
   frame, or Keepout proposes one from where the machine actually moves.
2. **Detects entry.** A person entering the zone while the machine is running
   raises an alert on the same frame, with the image that proves it.
3. **Checks the guard.** If a fixed guard is open, removed or covered, it says so.
   Machine Guarding (29 CFR 1910.212) is on OSHA's FY2024 top ten most-cited list.
4. **Escalates sensibly.** Brief entry with the machine stopped is a note. Entry
   while it is running is an alert. A person down and motionless is the highest
   level, and it **stays raised until a human acknowledges it**.

And, before any of that, it decides whether it can see at all. If the camera has
moved, the lens is blocked, the lights are out or the feed has frozen, it stops
answering and says which. A zone polygon means nothing against a view it was not
drawn on, and a silent "nobody in the zone" from a blind camera is the worst
failure this product can have.

## What it will not do

No face recognition. No identity, ever. Track numbers are per-run counters that
reset with the process and are never joined to a roster or a badge. There is no
code path that could add recognition without a new dependency.

Before an evidence frame is written, Keepout blurs the head region of **every
person it detects in that frame**. That includes people it is not tracking, found
by a low-threshold, tiled sweep of the raw frame, and every face the YuNet face
detector finds on top. The container refuses to start without the YuNet weights.
Each evidence record says which detectors ran. What this cannot promise: a person
that no detector finds at all is not blurred.

This claim used to be wrong. Until 16 September the README said "faces are blurred
in every stored evidence frame". Real construction footage showed that only the
incident's own subject was blurred, and the workers standing beside him stayed
sharp. [`docs/evaluation.md`](docs/evaluation.md) §10 has what happened and what
changed.

---

## Pinned dependencies

The one hard requirement of this competition is OpenCV 5, and there is a specific
trap: **OpenCV 4.14.0 was released after 5.0.0**, so an unpinned
`pip install opencv-python-headless` resolves to 4.x and the entry silently fails.

```
opencv-python-headless==5.0.0.93     # Apache-2.0
numpy==2.5.3
fastapi==0.141.1  starlette==1.6.0  pydantic==2.13.5
uvicorn[standard]==0.53.0  python-multipart==0.0.32
boto3==1.43.95                        # only when mirroring evidence to S3
```

Full list with transitive pins in [`requirements.txt`](requirements.txt).
`import visioncore` raises on a 4.x wheel, the Dockerfile asserts the major
version after install, and the running version is printed at `/version` and in the
top right of the UI.

**Detector:** YOLOX-tiny (Megvii, **Apache-2.0**), official ONNX export, run
through `cv2.dnn`. Nothing in this repository imports `ultralytics`, directly or
transitively: YOLOv5/v8/YOLO11 and FastSAM are AGPL-3.0, whose section 13 makes a
hosted demo API a source-disclosure event.

**Footage:** person imagery is matted from `samples/data/vtest.avi` in the OpenCV
repository, Apache-2.0, the same licence as the library. The real-footage check
used Wikimedia Commons clips under CC BY-SA 3.0 and CC BY 4.0, listed in
[`eval/real_footage/README.md`](eval/real_footage/README.md). None of them are bundled.

**Face detector:** YuNet, `face_detection_yunet_2023mar.onnx` from opencv_zoo, MIT,
fetched and checked against its sha256 at image build time. Run through
`cv2.FaceDetectorYN` only to decide what to blur.

---

## Run it

```bash
# from the repository root
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python \
  -e packages/visioncore -e packages/servicekit -e products/keepout

keepout build-samples                 # ~2 min: fetch, matte, render, label
# optional locally, required in the container: the YuNet face detector (MIT)
curl -fsSL -o ~/.cache/opencv26/models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
keepout demo --bake                   # ~4 min: analyse every clip once
uvicorn keepout.service:app --port 8000
```

Then open <http://127.0.0.1:8000>. To run your own footage, press **Upload a
video** in the sidebar. Draw the danger zone and the machine region on a frame of
it, check them over the picture in the live view, then run it.

`build-samples` needs `ffmpeg` on the path to re-encode the rendered clips to
H.264. Without it the clips are still analysable but browsers will not play them,
and the command says so rather than shipping a black demo.

### Command line

```bash
keepout build-samples [--force]   # stage the labelled evaluation clips
keepout analyse CLIP [--evidence DIR]
keepout demo --bake [--clip all]  # pre-compute what the service serves
keepout evaluate [--out DIR]      # score against the labelled set
keepout version                   # what is actually installed
```

---

## Test it

```bash
python -m pytest products/keepout/tests     # 143 tests, ~30 s
python -m ruff check products/keepout/src products/keepout/tests
```

The tests are mostly sequences whose answer is known by construction: a person
jittering on a zone boundary must produce one incident and not forty; a crouching
worker who keeps moving must not be reported as down; a moved camera must produce
no zone incidents at all. Three of them exist because they caught real bugs, and
the bug is named in the test. `tests/test_real_footage_regressions.py` holds one
test for each defect that real Wikimedia Commons footage exposed. Each is built by
construction and names the clip that showed it.

The Hungarian assignment is checked against brute-force enumeration on random
matrices, because that is the only way to know an assignment implementation is
correct.

---

## Deploy it

```bash
infra/s3.sh                                        # shared artefact bucket
infra/ecr.sh keepout --context . --dockerfile products/keepout/Dockerfile
infra/apprunner.sh keepout --tag <tag> --cpu 4 --memory 8 --port 8000 \
  --env KEEPOUT_EVIDENCE_BUCKET=opencv26-artifacts-<aws-account-id> \
  --env KEEPOUT_EVIDENCE_PREFIX=keepout/evidence \
  --env AWS_REGION=us-east-1
products/keepout/infra/instance-role.sh            # S3 write permission
```

Build from the **repository root**, not from this directory: the image needs
`packages/visioncore` and `packages/servicekit`.

See [`docs/costs.md`](docs/costs.md) for what was created, what it costs, and why
the service ended up in `eu-west-2` rather than `us-east-1`.

### Environment

| Variable | Default | What it does |
|---|---|---|
| `KEEPOUT_EVIDENCE_BUCKET` | unset | mirror evidence frames to this S3 bucket |
| `KEEPOUT_EVIDENCE_PREFIX` | `evidence` | key prefix within the bucket |
| `KEEPOUT_SAMPLES_DIR` | bundled | where the sample clips live |
| `KEEPOUT_DEMO_DIR` | bundled | where the baked runs live |
| `KEEPOUT_INSTANCE_LABEL` | `local workstation` | what the UI shows as the host |
| `KEEPOUT_MAX_FRAMES` | `900` | frames analysed per upload; a longer clip is covered by raising the stride, and the record says so |
| `KEEPOUT_YUNET_PATH` | unset | the YuNet face detector weights; also looked for in `OPENCV26_MODEL_DIR` and the model cache |
| `KEEPOUT_REQUIRE_FACE_DETECTOR` | unset (`1` in the container) | refuse to start without YuNet rather than fall back to head regions only |
| `OPENCV26_MODEL_DIR` | `~/.cache/opencv26/models` | ONNX weight cache |

---

## The endpoint

| Route | What it gives you |
|---|---|
| `GET /` | the control room |
| `GET /healthz` | liveness |
| `GET /version` | OpenCV version, build flags, git sha, model licences |
| `GET /api/keepout` | the escalation ladder, the privacy statement, the source quote |
| `GET /api/demo?clip=NAME` | a run pre-computed at image build time |
| `GET /api/samples` | the bundled clips and their ground truth |
| `GET /api/samples/{name}/frame` | a reference frame for drawing a zone on |
| `POST /api/zones/propose` | propose a zone from where the machine moves |
| `POST /api/jobs` | upload a clip and analyse it |
| `GET /api/jobs/{id}/events` | progress, as Server-Sent Events |
| `POST /api/jobs/{id}/incidents/{id}/acknowledge` | clear a latched critical |

Every bundled clip is analysed at **image build time** and served from disk, so a
cold container shows a real alert with its evidence frame in under two seconds.
"Run this clip live" re-runs the same code on the instance. That is not laziness:
App Runner's vCPUs run YOLOX-tiny at about 386 ms per frame against 14 ms on a
22-thread workstation, so a live run of a 24 second clip takes over a minute. The
number and the reasoning are in [`docs/evaluation.md`](docs/evaluation.md) §7.

---

## Results, briefly

Measured on 2,440 frames across seven clips, six with frame-level ground truth.

| | |
|---|---|
| Detection rate inside the zone | **819 / 819 frames**, 95% CI [0.9955, 1.0] |
| Time to alert | **0 ms**, median and maximum, on 4 of 4 labelled entries |
| Escalation level correct | **4 / 4** |
| False alerts, empty room | **0** in 0.0111 camera-hours |
| View problems detected | **4 / 4** kinds, 0 incidents raised while blind |
| Guard removal | detected in **1.5 s** |
| Person down | detected in **8.5 s** |
| Throughput | **26 ms/frame** on 22 threads, **440 ms/frame** on App Runner |

### On real footage

The numbers above come from staged clips. On 16 September Keepout was run over
real fixed-camera footage from Wikimedia Commons: a concrete-pump crew in Malta, a
time-lapse of a house going up under a crane, and an Amazon loading dock. It
exposed five defects. All five are fixed, and each has a regression test.

| | Before | After |
|---|---|---|
| Bystander faces in evidence frames | **sharp** | head region of every detected person blurred |
| House clip that opens on a black frame | **0** of 2,689 frames watched | 1,764 judged usable |
| False person-down on a bag at the frame edge | 1 | **0** |
| Alerts for about 9 people in the Malta pump zone | 27 | **16** |
| False criticals, courtyard walkway | 1 | **0** |
| False criticals, house time-lapse (crane zone) | 14 | **5** |

What is still wrong is in [`docs/evaluation.md`](docs/evaluation.md) §10. The
time-lapse still raises five false criticals, and we have not bent the rule to fit
time-lapse video. A hard cut to a similar-looking second camera left 165 of 1,085
frames judged usable. One YOLOX-tiny pass reliably finds people from about 15% of
frame height, tiled detection from about 8%. Every critical in the real-footage
runs was false, because nobody fell.

**No deployed-system outcome trial exists for this class of product.** If you ask
whether Keepout has been shown to prevent an injury in a real factory, the honest
answer is that nobody has published that evidence for any product in this
category.

---

## Documentation

| | |
|---|---|
| [`docs/report.md`](docs/report.md) | the technical report: problem, users, architecture, OpenCV 5 implementation, AWS, evaluation, limitations, responsible use |
| [`docs/architecture.md`](docs/architecture.md) | pipeline and AWS diagrams |
| [`docs/evaluation.md`](docs/evaluation.md) | every measured number and how to reproduce it |
| [`docs/costs.md`](docs/costs.md) | what was created on AWS and what it costs |
| [`docs/devpost.md`](docs/devpost.md) | submission text |
| [`docs/narration.md`](docs/narration.md) | the video script |
| [`docs/deck/`](docs/deck/) | four slides and an end card |

## Licence

Code in this directory is part of a competition entry and is not published under
an open-source licence. Third-party components keep their own: OpenCV Apache-2.0,
YOLOX Apache-2.0, YuNet MIT, `vtest.avi` Apache-2.0, NumPy BSD-3, FastAPI MIT.
