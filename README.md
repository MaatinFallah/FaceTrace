# FaceTrace

A Python app that watches a webcam feed, detects faces (even multiple at once, even turned to the side or tilted up/down), matches them against a local database, and lets you register new people on the spot. This was a learning project — the point was testing out my own ideas around face tracking and matching, not showcasing polished coding on its own.

## How this was built

The design is mine: the confirmation-before-registering logic, the multi-angle embedding idea, using a Ball Tree so lookups stay fast as the database grows, the cooldown system, the whole architecture. I used AI assistance to help write and debug the code that implements those ideas, so I want to be upfront that the code isn't 100% hand-typed by me line by line. The concepts, the research behind them, and the decisions about how everything should work are what I actually did here.

## What it actually does

- **Detects faces at odd angles.** Uses InsightFace's `buffalo_l` model, which works off 3D-aware landmarks, so a face looking sideways or tilted up/down still gets picked up and identified reliably — not just clean frontal shots.
- **Handles multiple people in frame at once.** Every detected face gets its own bounding box, its own match attempt, and its own status independently.
- **Recognizes people it already knows.** Shows their first + last name, a live similarity score, how many times they've been seen, and when they were last seen — right on the video feed.
- **Registers new people live.** When an unfamiliar face is confirmed, pressing `R` lets you type in a first and last name in the console, and it gets saved to the database — no restart needed.
- **Stays fast as the database grows.** Instead of comparing a new face to every stored face one by one, it builds a Ball Tree over all embeddings and does an approximate nearest-neighbor search, so lookups don't slow down linearly as more people get added.
- **GPU-accelerated when available.** Uses ONNX Runtime and automatically picks CUDA (NVIDIA) or DirectML (Windows) if present, falling back to CPU otherwise — no manual config needed.
- **Filters out false positives.** An unknown face has to show up for 15 consecutive frames before it's even offered up for registration, so a random blur or partial detection doesn't spam you with prompts.
- **Captures multiple angles per person.** On registration, instead of saving just one embedding, it picks the 5 *most different* embeddings it collected (via farthest-point sampling) — so the person is easier to recognize later even from a different angle than the one they registered with.
- **Avoids double-counting.** A recognized person's "seen count" only increments once every 30 seconds, even if they stand in frame the whole time.
- **Runs at a usable frame rate.** Inference only runs every 2nd frame (registration excluded), with FPS displayed live in the corner.

## Controls

| Key | Action |
|-----|--------|
| `R` | Register the currently pending unknown face |
| `S` | Skip/cancel a pending registration, or clear tracked unknown faces |
| `D` | Print the full database (names, times seen, embedding count) to the console |
| `Q` / `Esc` | Quit |

On screen, box color tells you the status at a glance: **green** = recognized, **red** = unknown/still being tracked, **orange** = confirmed and ready to register (or actively registering).

## Tech stack

- **Python 3**
- **OpenCV** — camera capture, drawing, display
- **InsightFace** (`buffalo_l`) — face detection + embeddings
- **ONNX Runtime** — runs the InsightFace model, with GPU support
- **scikit-learn** — Ball Tree for nearest-neighbor search
- **SQLite** — local storage, no server needed
- **NumPy** — embedding math

## Project structure

```
FaceTrace/
├── main.py                  # entry point, camera loop, keyboard controls
├── config.py                 # all tunable settings in one place
├── database.py                # SQLite layer (persons + embeddings)
├── face_engine.py            # InsightFace wrapper + Ball Tree search
├── recognition_manager.py    # ties detection, matching, and registration together
├── ui_manager.py              # all the OpenCV drawing/overlay code
├── schema.sql                 # database schema
└── data/                       # created on first run, holds faces.db (gitignored)
```

## Getting started

```bash
git clone https://github.com/<your-username>/FaceTrace.git
cd FaceTrace
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

You'll need a working webcam. On first run, InsightFace will automatically download the `buffalo_l` model (roughly 300 MB) to `~/.insightface` — this only happens once.

## Configuration

Everything tunable lives in `config.py`:

- `SIMILARITY_THRESHOLD` — how close an embedding needs to be to count as a match (default `0.42`)
- `RECOGNITION_COOLDOWN_SECONDS` — minimum gap before re-counting a known person (default `30`)
- `UNKNOWN_CONFIRMATION_FRAMES` — consecutive frames needed before offering to register a new face (default `15`)
- `CAMERA_INDEX`, `FRAME_WIDTH`, `FRAME_HEIGHT` — camera settings

## Database

Two tables: `person` (name, first/last seen timestamps, recognition count) and `face_embedding` (multiple embeddings per person, each labeled by angle — `primary`, `angle_1`, etc.). A person can have several embeddings; deleting a person cascades and removes theirs too.

## Limitations

Being upfront about what this doesn't do: there's no liveness/anti-spoofing check, so a photo of someone's face could in theory fool it. It's single-camera only, and accuracy drops in poor lighting like any vision model. It's a learning project, not a production security tool.

## Possible next steps

- Basic liveness detection (blink/movement check) to prevent photo spoofing
- Export/import the database so it's portable between machines
- A proper GUI instead of the OpenCV window
- Multi-camera support

## A note on privacy

This app stores real face embeddings and names locally in `data/faces.db`. That folder is gitignored on purpose — don't commit your own recognition database to a public repo.

## License

MIT — see [LICENSE](LICENSE).
