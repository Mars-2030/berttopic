# Deployment Guide

This guide covers various deployment options for the Social Media Topic Modeling System.

## Local Development

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

### Development with Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t topic-modeling-app .
docker run -p 8501:8501 topic-modeling-app
```

## Production Deployment

### Docker Production Setup

1. **Build the production image:**
```bash
docker build -t topic-modeling-app:latest .
```

2. **Run with production settings:**
```bash
docker run -d \
  --name topic-modeling-prod \
  -p 8501:8501 \
  --memory=4g \
  --cpus=2 \
  --restart=unless-stopped \
  topic-modeling-app:latest
```

3. **Using Docker Compose for production:**
```yaml
version: '3.8'
services:
  topic-modeling-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Cloud Deployment Options

#### 1. AWS ECS/Fargate
```bash
# Tag for ECR
docker tag topic-modeling-app:latest your-account.dkr.ecr.region.amazonaws.com/topic-modeling-app:latest

# Push to ECR
docker push your-account.dkr.ecr.region.amazonaws.com/topic-modeling-app:latest
```

#### 2. Google Cloud Run
```bash
# Build and deploy to Cloud Run
gcloud run deploy topic-modeling-app \
  --image gcr.io/your-project/topic-modeling-app \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2
```

#### 3. Azure Container Instances
```bash
# Deploy to Azure
az container create \
  --resource-group myResourceGroup \
  --name topic-modeling-app \
  --image your-registry.azurecr.io/topic-modeling-app:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8501
```

#### 4. Heroku
```bash
# Login to Heroku Container Registry
heroku container:login

# Build and push
heroku container:push web --app your-app-name

# Release
heroku container:release web --app your-app-name
```

### Kubernetes Deployment

#### Deployment YAML
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: topic-modeling-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: topic-modeling-app
  template:
    metadata:
      labels:
        app: topic-modeling-app
    spec:
      containers:
      - name: topic-modeling-app
        image: topic-modeling-app:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: STREAMLIT_SERVER_PORT
          value: "8501"
        - name: STREAMLIT_SERVER_ADDRESS
          value: "0.0.0.0"
---
apiVersion: v1
kind: Service
metadata:
  name: topic-modeling-service
spec:
  selector:
    app: topic-modeling-app
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

## Performance Optimization

### Memory Management
- **Minimum RAM**: 4GB for small datasets (< 1000 documents)
- **Recommended RAM**: 8GB+ for larger datasets
- **Large datasets**: Consider processing in batches

### CPU Optimization
- **Minimum**: 2 CPU cores
- **Recommended**: 4+ CPU cores for faster processing
- **GPU**: Optional, can speed up transformer models

### Storage Considerations
- **Docker image**: ~2GB
- **Temporary files**: Varies with dataset size
- **Persistent storage**: Optional for saving results

## Monitoring and Logging

### Health Checks
The application includes built-in health checks:
```bash
# Check application health
curl http://localhost:8501/_stcore/health
```

### Logging
Streamlit logs are available through Docker:
```bash
# View logs
docker logs topic-modeling-app

# Follow logs
docker logs -f topic-modeling-app
```

### Monitoring with Prometheus
Add monitoring endpoints for production:
```python
# Add to streamlit_app.py for monitoring
import time
import psutil

# Add metrics endpoint
@st.cache_data
def get_system_metrics():
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'timestamp': time.time()
    }
```

## Security Considerations

### Container Security
- Run as non-root user (included in Dockerfile)
- Use minimal base images
- Regularly update dependencies

### Network Security
- Use HTTPS in production
- Implement proper firewall rules
- Consider VPN for internal access

### Data Security
- Encrypt data at rest and in transit
- Implement proper access controls
- Regular security audits

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Increase container memory limits
   - Process smaller datasets
   - Use batch processing

2. **Slow Performance**
   - Increase CPU allocation
   - Use SSD storage
   - Optimize dataset size

3. **Container Won't Start**
   - Check logs: `docker logs container-name`
   - Verify port availability
   - Check resource limits

4. **Model Loading Issues**
   - Ensure internet connectivity for model downloads
   - Pre-download models in Docker build
   - Check disk space

### Support
For deployment issues:
1. Check the logs first
2. Verify system requirements
3. Test with sample data
4. Check network connectivity

