# Safety and security

## Safety guarantees

- **Immich's Docker image is unmodified.** No custom images, no patches.
- **The native worker runs Immich's own code.** Extracted from the Docker image, not reimplemented.
- **UPSERT-safe database writes.** The native worker uses Immich's own job pipeline with the same UPSERT logic.
- **Version-matched.** The extracted server always matches the Docker image version exactly.

See [../README.md#what-we-modify-and-how-to-undo-it](../README.md#what-we-modify-and-how-to-undo-it) for exactly what's changed on disk and how to revert it.

## Network-facing surfaces

| Surface | Default bind | Auth | Notes |
|---------|--------------|------|-------|
| Config file (`~/.immich-accelerator/config.json`) | n/a (local file) | filesystem | chmod 600 |
| Postgres | `127.0.0.1:5432` | Postgres auth | localhost only by default |
| Redis | `127.0.0.1:6379` | none (Redis default) | localhost only by default |
| Dashboard | `0.0.0.0:8420` | none | LAN-accessible. The Re-queue button triggers job processing via the Immich API — see [usage.md#dashboard](usage.md#dashboard). Set `"dashboard_host": "127.0.0.1"` to keep it on this Mac. |
| ML service (port 3003) | `0.0.0.0:3003` | none | LAN-accessible, same trust model as Immich's own Docker `machine-learning` container — see [deployment.md#security-note](deployment.md#security-note). Set `"ml_host": "127.0.0.1"` only if the worker runs on this Mac; Immich in Docker reaches it through `host.docker.internal`, which is not loopback. |

Neither the dashboard nor the ML service authenticates its caller. That is
fine on a network you control and wrong on one you do not: the dashboard's
`/api/requeue` and `/api/auto-requeue` accept a POST from any client that can
reach the port (its same-origin check stops a browser tab, not `curl`), and
`/predict` accepts images from anything that can open a TCP connection.

### Keeping them off the network

Both bind every interface by default, which is what lets a phone on the same
LAN watch progress and what lets Immich in Docker reach the ML engine. To
restrict either one to this Mac, set its host in `config.json` and restart:

```json
{
  "dashboard_host": "127.0.0.1",
  "ml_host": "127.0.0.1"
}
```

Only those two values are accepted (`0.0.0.0` for every interface, `127.0.0.1`
for loopback). Every liveness check the accelerator runs probes loopback, so a
listener bound to one specific LAN address would look dead to the supervisor
and be started twice; other values are ignored with a warning rather than
half-supported. The keys survive `setup` re-runs and upgrades.

`"ml_host": "127.0.0.1"` requires the worker to run on this Mac. A split
deployment whose Immich is in Docker on another machine (or in Docker on this
one) connects through `host.docker.internal`, which cannot reach a loopback
listener: leave `ml_host` at its default there.

If you need the dashboard beyond your LAN, put it behind a firewall or VPN
rather than exposing it to the internet.
