# Whisper Large Turbo STT Server

This directory contains the Docker Compose configuration for running a Whisper STT (Speech-to-Text) server using the `faster-whisper-large-v3-turbo` model with CUDA acceleration.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed

## Configuration

The server uses the following configuration:
- **Model**: `Systran/faster-whisper-large-v3-turbo-ct2`
- **Compute Type**: `float16` (best accuracy/speed balance on GPU)
- **Device**: `cuda` (GPU acceleration)
- **Port**: `8030`

## Deployment

### Start the server

```bash
docker-compose up -d
```

### View logs

```bash
docker-compose logs -f whisper-server
```

### Stop the server

```bash
docker-compose down
```

## Cache

The Hugging Face cache is persisted in `./hf_cache` to avoid re-downloading the 2GB+ model on every restart.

## API Usage

Once running, the Whisper server will be available at `http://localhost:8030`.

### Example API call

```bash
curl -X POST "http://localhost:8030/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.wav" \
  -F "model=Systran/faster-whisper-large-v3-turbo-ct2"
```

## Troubleshooting

### Check if the container is running

```bash
docker ps | grep whisper_server
```

### Check GPU availability

```bash
docker exec whisper_server nvidia-smi
```

### Restart the server

```bash
docker-compose restart
```
