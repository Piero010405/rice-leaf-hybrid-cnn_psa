# Docker

El proyecto puede ejecutarse localmente o con Docker.

GPU:

```bash
docker compose -f docker/docker-compose.gpu.yml build
docker compose -f docker/docker-compose.gpu.yml run --rm riceleaf-gpu python scripts/00_check_environment.py
```
