# Keepout

**A fixed camera over machinery, watched by something that never blinks.**

Live: <https://mqrmfp6mbi.eu-west-2.awsapprunner.com>

An operator draws a danger zone once on a reference frame. A person entering it
while the machine is running raises an alert on that same frame, with the picture
that proves it. A person down and motionless raises a level that stays up until a
human acknowledges it. And before any of that, four checks decide whether the view
can be trusted at all — if the camera has moved, the lens is blocked, the lights are
out or the feed has frozen, Keepout names the problem and measures nothing.

## Results

Measured on 2,440 frames across seven clips, six with frame-level ground truth.
Method and reproduction: [`docs/evaluation.md`](docs/evaluation.md).

| | |
|---|---|
| Frames with somebody in the zone, seen | **819 / 819**, 95% CI [0.9955, 1.0] |
| Time to alert | **0 ms**, median and maximum, on 4 of 4 labelled entries |
| Escalation level correct | **4 / 4** |
| Incidents raised while blind | **0** across 228 frames |
| View problems detected | **4 / 4** kinds |
| Guard removal | detected in **1.5 s** |
| Person down | detected in **8.5 s** |
| False alerts, empty room | **0** in 0.0111 camera-hours |
| Throughput | **26 ms/frame** on 22 threads, **440 ms/frame** on App Runner |

The labelled set is **composited**: real people matted out of OpenCV's `vtest.avi`
pedestrian clip, placed into a rendered machine cell, so the truth is known exactly.

On real fixed-camera construction footage from Wikimedia Commons, which carries no
labels and so counts false alarms rather than misses: the Malta pump crew raises 16
alerts for about nine people in the zone, 1.8 per person, with no false criticals;
the house time-lapse with the crane zone raises 13 alerts and 5 false criticals,
which stay, because one second of that video is a minute of real time; a handheld
warehouse clip is refused almost entirely as `camera_moved`. Nobody fell in any of
them, so every critical they raise is false.

**No deployed-system outcome trial exists for this class of product.** Asked whether
Keepout has been shown to prevent an injury in a real factory, the honest answer is
that nobody has published that evidence for any product in this category.

---

## What it does

1. **Learns the danger zone.** Drawn once against a reference frame, or proposed
   from where the machine actually moves.
2. **Detects entry.** A person in the zone while the machine runs raises an alert on
   the same frame, with the image that proves it.
3. **Checks the guard.** If a fixed guard is open, removed or covered, it says so.
   Machine Guarding (29 CFR 1910.212) is on OSHA's FY2024 top ten most-cited list.
4. **Escalates sensibly.** Entry with the machine stopped is a note, because a
   changeover is routine work. Entry while it runs is an alert. A person down and
   motionless is the highest level, and it **stays raised until a human acknowledges
   it**.

A zone polygon means nothing against a view it was not drawn on, and a silent
"nobody in the zone" from a blind camera is the worst failure this product can have.
That is why the view checks come first and why "incidents raised while blind" is a
first-class number.

## What it will not do

No face recognition. No identity, ever. Track numbers are per-run counters that
reset with the process and are never joined to a roster or a badge. There is no code
path that could add recognition without a new dependency.

Before an evidence frame is written, Keepout blurs the head region of **every person
it detects in that frame** — including people it is not tracking, found by a
low-threshold tiled sweep of the raw frame — and every face the YuNet face detector
finds on top. The container refuses to start without the YuNet weights, and each
evidence record says which detectors ran. What this cannot promise: a person that no
detector finds at all is not blurred.

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
`import visioncore` raises on a 4.x wheel, the Dockerfile asserts the major version
after install, and the running version is printed at `/version` and in the UI.

**Detector:** YOLOX-tiny (Megvii, **Apache-2.0**), official ONNX export, through
`cv2.dnn`. Nothing here imports `ultralytics`, directly or transitively: YOLOv5/v8/
YOLO11 and FastSAM are AGPL-3.0, whose section 13 makes a hosted demo API a
source-disclosure event.

**Face detector:** YuNet, `face_detection_yunet_2023mar.onnx` from opencv_zoo, MIT,
fetched and checked against its sha256 at image build time. Run through
`cv2.FaceDetectorYN` only to decide what to blur.

**Footage:** person imagery is matted from `samples/data/vtest.avi` in the OpenCV
repository, Apache-2.0. The real-footage check used Wikimedia Commons clips under
CC BY-SA 3.0 and CC BY 4.0, listed in
[`eval/real_footage/README.md`](eval/real_footage/README.md). None are bundled.

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

Then open <http://127.0.0.1:8000>. To run your own footage, press **Upload a video**
in the sidebar, draw the danger zone and the machine region on a frame of it, check
them over the picture in the live view, then run it.

`build-samples` needs `ffmpeg` on the path to re-encode the rendered clips to H.264.
Without it the clips are still analysable but browsers will not play them, and the
command says so rather than shipping a black demo.

```bash
keepout build-samples [--force]   # stage the labelled evaluation clips
keepout analyse CLIP [--evidence DIR]
keepout demo --bake [--clip all]  # pre-compute what the service serves
keepout evaluate [--out DIR]      # score against the labelled set
keepout version                   # what is actually installed
```

## Test it

```bash
python -m pytest products/keepout/tests     # 143 tests, ~30 s
python -m ruff check products/keepout/src products/keepout/tests
```

The tests are mostly sequences whose answer is known by construction: a person
jittering on a zone boundary must produce one incident and not forty; a crouching
worker who keeps moving must not be reported as down; a moved camera must produce no
zone incidents at all. The Hungarian assignment is checked against brute-force
enumeration on random matrices, because that is the only way to know an assignment
implementation is correct. `tests/test_real_footage_regressions.py` holds one test
for each defect that real Wikimedia Commons footage exposed, built by construction
and naming the clip that showed it.

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
`packages/visioncore` and `packages/servicekit`. The service runs in `eu-west-2`
because the account is limited to two App Runner services per region.

| Variable | Default | What it does |
|---|---|---|
| `KEEPOUT_EVIDENCE_BUCKET` | unset | mirror evidence frames to this S3 bucket |
| `KEEPOUT_EVIDENCE_PREFIX` | `evidence` | key prefix within the bucket |
| `KEEPOUT_SAMPLES_DIR` | bundled | where the sample clips live |
| `KEEPOUT_DEMO_DIR` | bundled | where the baked runs live |
| `KEEPOUT_MAX_FRAMES` | `900` | frames analysed per upload; a longer clip is covered by raising the stride, and the record says so |
| `KEEPOUT_YUNET_PATH` | unset | the YuNet weights; also looked for in `OPENCV26_MODEL_DIR` and the model cache |
| `KEEPOUT_REQUIRE_FACE_DETECTOR` | unset (`1` in the container) | refuse to start without YuNet rather than fall back to head regions only |
| `OPENCV26_MODEL_DIR` | `~/.cache/opencv26/models` | ONNX weight cache |

## The endpoint

| Route | What it gives you |
|---|---|
| `GET /` | the control room |
| `GET /healthz` | liveness |
| `GET /version` | OpenCV version, build flags, git sha, model licences |
| `GET /api/keepout` | the escalation ladder and the privacy statement |
| `GET /api/demo?clip=NAME` | a run pre-computed at image build time |
| `GET /api/samples` | the bundled clips and their ground truth |
| `GET /api/samples/{name}/frame` | a reference frame for drawing a zone on |
| `POST /api/zones/propose` | propose a zone from where the machine moves |
| `POST /api/jobs` | upload a clip and analyse it |
| `GET /api/jobs/{id}/events` | progress, as Server-Sent Events |
| `POST /api/jobs/{id}/incidents/{id}/acknowledge` | clear a latched critical |

Every bundled clip is analysed at **image build time** and served from disk, so a
cold container shows a real alert with its evidence frame in under two seconds. "Run
this clip live" re-runs the same code on the instance. App Runner's vCPUs run
YOLOX-tiny at about 386 ms per frame against 14 ms on a 22-thread workstation, so a
live run of a 24 second clip takes over a minute ([`docs/evaluation.md`](docs/evaluation.md)
section 7).

## Documentation

| | |
|---|---|
| [`docs/report.md`](docs/report.md) | the technical report: problem, users, architecture, OpenCV 5 implementation, AWS, evaluation, limitations, responsible use |
| [`docs/architecture.md`](docs/architecture.md) | pipeline and AWS diagrams |
| [`docs/evaluation.md`](docs/evaluation.md) | every measured number and how to reproduce it |

## Licence

Code in this directory is released under the MIT licence (see `LICENSE` at the
repository root). Third-party components keep their own: OpenCV Apache-2.0, YOLOX
Apache-2.0, YuNet MIT, `vtest.avi` Apache-2.0, NumPy BSD-3, FastAPI MIT.
