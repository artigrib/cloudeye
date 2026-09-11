# Install

Gets you a running API, worker and web UI on one machine. No GPU is needed for this — video
processing is, everything else is not.

Reproduced end to end in a fresh directory against a throwaway database before each release
of these docs.

## Requirements

- Ubuntu 24.04 (or any Linux with the packages below)
- Python **3.12** and [uv](https://docs.astral.sh/uv/)
- PostgreSQL **16**, running natively
- Redis, running natively, on `127.0.0.1:6379`
- Node **22** and npm
- `ffmpeg` (it provides `ffprobe`, which the upload path uses for video metadata)

```bash
sudo apt-get install -y postgresql-16 redis-server ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 1. Clone and configure

```bash
git clone https://github.com/artigrib/cloudeye.git
cd cloudeye
cp .env.example .env
```

Open `.env`. Every variable is documented in place. The three that matter now:

- `DATABASE_URL` — must match the role and database you create in step 2.
- `REDIS_URL` — the default is right if Redis is local.
- `UPLOAD_DIR` — defaults to `./var/uploads`, inside the clone. Point it at a real data
  volume for anything but a trial.

`var/` is gitignored and is where every out-of-repo tree is mounted or symlinked — uploads,
scratch, assets, nvblox output. Create what you need:

```bash
mkdir -p var/uploads var/frontend_layers
```

## 2. Database

```bash
sudo -u postgres createuser robot --pwprompt      # use 'robot' to match .env.example
sudo -u postgres createdb robotdb -O robot
```

Schema is Alembic-managed — never `create_all`:

```bash
uv sync
uv run alembic upgrade head
uv run alembic current          # should print the head revision
```

## 3. Run it

Three processes. Separate terminals, or `systemd` — see [DEPLOY.md](DEPLOY.md).

```bash
./run_api.sh                                  # API on 127.0.0.1:8000
uv run arq app.worker.WorkerSettings          # the job worker
cd frontend && npm ci && npm run dev          # web UI on 127.0.0.1:5173
```

Open <http://127.0.0.1:5173/workspaces>.

The API binds loopback only, on purpose. To reach it from elsewhere, tunnel:
`ssh -L 5173:127.0.0.1:5173 -L 8000:127.0.0.1:8000 <host>`.

### Check it

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health         # 200
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/api/projects   # 200
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:5173/workspaces/    # 200
```

`GET /workspaces/` is a **UI route**, served by the dev server — that is what returns 200.
There is no `GET /workspaces/` API endpoint: the API mounts everything under `/api`, the
workspace list is `GET /api/projects`, and the only route under `/api/workspaces` is
`GET /api/workspaces/{id}/processing`.

### Running against a throwaway backend

The dev server proxies `/api` to `http://127.0.0.1:8000` by default. Override it so a
second stack never touches a running one:

```bash
VITE_API_PROXY_TARGET=http://127.0.0.1:8020 npx vite --port 5180 --strictPort
```

## 4. Optional: keys

Without an `OPENROUTER_API_KEY` the app runs, but the object-vocabulary stage and the chat
command box fail loudly rather than degrading — they do not invent a vocabulary. Get a key
at <https://openrouter.ai/keys> and put it in `.env`, or in a separate file that
`run_api.sh` sources (`$HOME/.secrets/openrouter.env` by default).

Vertex AI is an optional second provider for the vocabulary stage — see
[GOOGLE_CLOUD.md](GOOGLE_CLOUD.md). OpenRouter is the default; no Google setup is needed.

## 5. Processing video needs a GPU box

Everything above runs without one. To actually turn a video into a scene you need a rented
GPU instance reachable over ssh as the alias in `GPU_SSH_HOST`, with the three venvs the
stages expect. See [GPU_VAST.md](GPU_VAST.md) and [MODELS.md](MODELS.md).

## Tests

```bash
uv run pytest                     # pure unit: no DB, no GPU, no network
cd frontend && npm test           # vitest
python3 pipeline/tests/run_tests.py
```

The pipeline suite shadows `vastai`, `ssh`, `scp` and `rsync` with fakes, so it rents
nothing and touches no network. It does need fixtures that are **not in git** — a packed
depth stack, an nvblox output fixture and a pass-A bundle. Point it at them:

```bash
PIPELINE_TEST_FIXTURES=var/sample-scene python3 pipeline/tests/run_tests.py
```

Without them it exits 2 with a one-line reason naming what is missing. See
[GPU_VAST.md](GPU_VAST.md#sample-scene) for where to get a sample scene.

Three pytest cases are red on purpose, with reasons — see
[KNOWN_TEST_FAILURES.md](KNOWN_TEST_FAILURES.md). They are not a broken install.

## Docker

`docker-compose.yml`, `Dockerfile.api` and `Dockerfile.web` are in the repository and cover
postgres, redis, the API and the dev server.

> **Written, not executed.** These files have never been run on the machine this project is
> developed on — it is a live host where installing a container runtime was not acceptable.
> They are offered as a starting point and are **untested**. The native path above is the
> one that is verified.
