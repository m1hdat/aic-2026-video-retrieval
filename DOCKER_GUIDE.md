# Docker Guide

This project uses Docker Compose to run Milvus Standalone and its required services locally.

## What Is Docker?

Docker is a platform for packaging and running applications in isolated environments called containers.

A container includes the application and the runtime dependencies it needs, such as system libraries, service configuration, and startup commands. This makes the application easier to run consistently across different machines.

In a traditional setup, you might need to install Milvus, etcd, and MinIO directly on your computer. With Docker, this project can start those services from predefined images instead.

In this project, Docker helps with:

- Running Milvus without a manual Milvus installation.
- Keeping Milvus services separate from the Python environment.
- Making the setup easier to reproduce on another machine.
- Starting and stopping infrastructure with simple commands.

## What Is Docker Compose?

Docker Compose is a tool for running multiple containers from one configuration file.

This project uses `docker-compose.yml` to define all services required by Milvus. Instead of starting each service manually, you can run:

```bash
docker compose up -d
```

Docker Compose will start the full local Milvus stack.

## What Docker Runs

The `docker-compose.yml` file starts three services:

- `etcd`: stores Milvus metadata.
- `minio`: stores Milvus object data.
- `milvus`: the vector database used by the retrieval system.

Milvus listens on:

```text
localhost:19530
```

That is the same host and port configured in `configs/config.yaml`.

## Install Docker Desktop

On Windows, install Docker Desktop from the official Docker website:

```text
https://www.docker.com/products/docker-desktop/
```

After installation:

1. Start Docker Desktop.
2. Wait until Docker Desktop says the engine is running.
3. Open a new terminal.
4. Check Docker:

```bash
docker --version
docker compose version
```

If `docker` is not recognized, restart the terminal or restart Windows so the Docker CLI is added to `PATH`.

## Start Milvus

From the project root:

```bash
docker compose up -d
```

Check running containers:

```bash
docker compose ps
```

Expected services:

```text
etcd
minio
milvus
```

## View Logs

To inspect all service logs:

```bash
docker compose logs
```

To inspect only Milvus logs:

```bash
docker compose logs milvus
```

To follow logs live:

```bash
docker compose logs -f milvus
```

## Stop Services

Stop containers without deleting stored data:

```bash
docker compose down
```

Start them again later:

```bash
docker compose up -d
```

## Remove Milvus Data

Milvus data is stored under:

```text
outputs/milvus/
```

To fully reset Docker containers and volumes created by this compose file:

```bash
docker compose down
```

Then delete `outputs/milvus/` manually if you want a completely clean Milvus storage directory.

For most project resets, prefer the Python reset script instead:

```bash
python scripts/05_reset_collection.py
```

That resets only the Milvus collection and keeps the Docker services running.

## Common Problems

### `docker` is not recognized

Docker Desktop is not installed, not running, or its CLI is not in `PATH`.

Fix:

1. Install Docker Desktop.
2. Start Docker Desktop.
3. Open a new terminal.
4. Run:

```bash
docker --version
```

### Docker Desktop is running but containers do not start

Check logs:

```bash
docker compose logs
```

Also make sure ports `19530`, `9000`, `9001`, and `9091` are not already used by another service.

### Milvus connection fails from Python

Make sure containers are running:

```bash
docker compose ps
```

Then check that `configs/config.yaml` uses:

```yaml
milvus:
  host: localhost
  port: 19530
```
