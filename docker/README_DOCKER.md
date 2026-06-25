# Uso con Docker

## GPU

Requiere Docker, driver NVIDIA y NVIDIA Container Toolkit.

```bash
docker compose -f docker/docker-compose.gpu.yml build
docker compose -f docker/docker-compose.gpu.yml run --rm riceleaf-gpu python scripts/00_check_environment.py
```

## CPU

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm riceleaf-cpu python scripts/00_check_environment.py
```
