# Deploy

One Ubuntu 24.04 host runs the API, the worker, PostgreSQL and Redis. GPU work happens on a
rented box reached over ssh — nothing inbound is ever opened for it.

Assumes [INSTALL.md](INSTALL.md) has been done once by hand and works.

## Layout

| Thing | Where |
|---|---|
| Code | `/opt/cloudeye` (the unit files' `WorkingDirectory`) |
| Config | `/opt/cloudeye/.env` |
| Secrets | `/etc/cloudeye/openrouter.env`, mode 600 |
| Uploads and scene artifacts | whatever `UPLOAD_DIR` points at — put it on a data volume |
| `uv` | `/usr/local/bin/uv` |

```bash
sudo install -d -m 755 /opt/cloudeye
sudo install -d -m 700 /etc/cloudeye
sudo git clone https://github.com/artigrib/cloudeye.git /opt/cloudeye
cd /opt/cloudeye && sudo cp .env.example .env && sudo $EDITOR .env
printf 'OPENROUTER_API_KEY=...\n' | sudo tee /etc/cloudeye/openrouter.env >/dev/null
sudo chmod 600 /etc/cloudeye/openrouter.env
```

## Ports

| Port | Bind | What |
|---|---|---|
| 8000 | `127.0.0.1` | the API |
| 5432 | `127.0.0.1` | PostgreSQL |
| 6379 | `127.0.0.1` | Redis |
| 22 | public | ssh — the only thing that should be reachable |

`deploy/video-api.service` binds uvicorn to `127.0.0.1`. The comment above that line —
*"Loopback only - this must never be exposed on 0.0.0.0"* — is load-bearing, not decoration:
**this project has no authentication of any kind.** No login, no token, no API key check.
Anyone who can reach port 8000 can read every scene and delete any of them.

There is no reverse proxy, no TLS and no nginx config in this repository. Reach the UI over
an ssh tunnel:

```bash
ssh -L 5173:127.0.0.1:5173 -L 8000:127.0.0.1:8000 <host>
```

If you put a proxy in front of it instead, put authentication in the proxy first.

## Firewall (ufw)

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw enable
sudo ufw status verbose
```

That is the whole policy: deny inbound, allow ssh. Nothing else needs to be opened —
8000/5432/6379 are loopback, and the GPU box is reached *outbound* over ssh, so no inbound
port or webhook is needed for it.

Do not install a container runtime on this host without checking what it does to your
firewall: Docker inserts its own iptables chains and published ports can bypass ufw rules.

## systemd

Two units, both in `deploy/`. Edit the paths if yours differ, then:

```bash
sudo cp deploy/video-api.service deploy/video-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now video-api video-worker
systemctl status video-api video-worker --no-pager
```

Verify before you trust it:

```bash
sudo systemd-analyze verify /etc/systemd/system/video-api.service
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health   # 200
journalctl -u video-worker -n 50 --no-pager
```

**`EnvironmentFile=/opt/cloudeye/.env` has no `-` prefix, deliberately.** Every setting has
a hardcoded fallback in `app/config.py`, so a missing or unreadable `.env` must fail the
unit loudly instead of silently starting against `robot:robot@127.0.0.1/robotdb`. The
OpenRouter file keeps its `-`, since its absence only disables VLM calls.

`video-worker.service` also carries the two GPU-spend switches,
`PIPELINE_WORKER_ENABLED` and `PIPELINE_DAILY_USD`, **commented out**. Renting is opt-in.
Read [GPU_VAST.md](GPU_VAST.md) before uncommenting either, and remember that editing the
copy in the repository changes nothing until it is copied to `/etc/systemd/system` and
`daemon-reload`ed.

## Updating

**Apply migrations before restarting, every time.** A restart picks up whatever ORM models
are on disk immediately. If a model has a column the database does not have yet, every query
touching that table starts throwing `UndefinedColumnError` — this took production down for
about twelve minutes once, on code that had been running fine for hours, because the restart
was what exposed a migration nobody had applied. A pending migration is harmless only for as
long as nothing restarts.

```bash
cd /opt/cloudeye
sudo git pull
sudo -u root /usr/local/bin/uv sync --project /opt/cloudeye

# 1. Are we behind?
uv run alembic current
uv run alembic heads
# 2. If they differ, migrate FIRST.
uv run alembic upgrade head
# 3. Only now restart.
sudo systemctl restart video-api video-worker
# 4. Confirm.
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health
journalctl -u video-api -u video-worker -n 30 --no-pager
```

If the unit files themselves changed, `cp` them again and `systemctl daemon-reload` before
the restart.

### Rolling back

`git checkout <previous-tag> && systemctl restart video-api video-worker` is safe **only if
no migration ran**. Once one has, roll the schema back first with
`uv run alembic downgrade <rev>`, or the old code will meet a newer schema. Check
`alembic history` before assuming a downgrade exists.

## Backups

The database and `UPLOAD_DIR` are the state; the clone is replaceable.

```bash
sudo -u postgres pg_dump robotdb | zstd -o /backup/robotdb-$(date +%F).sql.zst
tar --zstd -cf /backup/uploads-$(date +%F).tar.zst -C "$UPLOAD_DIR" .
```

Scene artifacts under `UPLOAD_DIR` are regenerable by re-running the pipeline, at the cost
of GPU time — the database rows referencing them are not.
