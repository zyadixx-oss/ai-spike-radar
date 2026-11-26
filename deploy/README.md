# VPS deployment

Option A — Docker (recommended): see the root README, `docker compose up -d --build`.

Option B — systemd:
1. Install Python 3.11+, Node 20+, PostgreSQL (optional; SQLite works for one process).
2. Follow the commands at the top of `spike-radar.service`, then copy it to `/etc/systemd/system/`.
3. Create `/opt/spike-radar/.env` from `.env.example` (set `DATABASE_URL=postgresql://user:pass@localhost/spike_radar` for Postgres).
4. `sudo systemctl enable --now spike-radar` and check `journalctl -u spike-radar -f`.
5. Verify the real Deriv connection: `cd /opt/spike-radar/backend && .venv/bin/python -m app.cli probe`.
6. Put Caddy (`Caddyfile`) or Nginx in front for TLS.
