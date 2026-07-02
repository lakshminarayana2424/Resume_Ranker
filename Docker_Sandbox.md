# Docker Sandbox

A prebuilt Docker image is available on Docker Hub.

Docker Hub Repository

https://hub.docker.com/r/lakshminarayana2424/resume_ranker

Pull the image

```bash
docker pull lakshminarayana2424/resume_ranker:latest
```

Run the ranking pipeline

```bash
docker run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  lakshminarayana2424/resume_ranker:latest
```

The generated submission file will be written to:

```
outputs/submission.csv
```

This Docker image reproduces the CPU-only ranking pipeline described in this repository.