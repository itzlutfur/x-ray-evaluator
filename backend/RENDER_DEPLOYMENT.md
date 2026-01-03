# Deploying to Render.com

## Prerequisites

1. Push your code to a Git repository (GitHub, GitLab, or Bitbucket)
2. Sign up at [render.com](https://render.com)
3. Handle model file storage (see below)

## Model Files Storage

⚠️ **CRITICAL:** Your `.keras` model files are too large to include in the Git repository and may exceed Render's build size limits.

### Option 1: External Storage (Recommended)
Store models in cloud storage and download during startup:
- **AWS S3:** Use boto3 to download files
- **Google Cloud Storage:** Use google-cloud-storage
- **Public URL:** Use curl/wget in `build.sh`

### Option 2: Render Persistent Disk (Paid)
1. Uncomment the `disk` section in `render.yaml`
2. Manually upload models to the mounted disk via SSH or deploy script

### Option 3: Git LFS (if models < 2GB total)
1. Install Git LFS: `git lfs install`
2. Track model files: `git lfs track "*.keras"`
3. Commit and push

## Deployment Steps

### Method 1: Using render.yaml (Recommended)

1. **Push your code** to GitHub/GitLab
2. **Go to Render Dashboard** → "New" → "Blueprint"
3. **Connect repository** and select your repo
4. Render will automatically detect `render.yaml` in the `backend` directory
5. **Configure Environment Variables** in the dashboard:
   - Update `XRAY_CORS_ORIGINS` with your actual frontend URL
   - Add any other required variables
6. **Deploy!**

### Method 2: Manual Web Service Setup

1. **Go to Render Dashboard** → "New" → "Web Service"
2. **Connect repository** and select your repo
3. **Configure:**
   - **Name:** `xray-evaluator-backend`
   - **Root Directory:** `backend`
   - **Environment:** `Python 3`
   - **Build Command:** `bash build.sh` or `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free or paid tier

4. **Environment Variables:**
   ```
   PYTHON_VERSION=3.11.0
   XRAY_CORS_ORIGINS=["https://your-frontend.com"]
   XRAY_MODEL_DIR=/opt/render/project/src/backend/models
   ```

5. **Deploy!**

## Post-Deployment

1. **Check Health:** Visit `https://your-service.onrender.com/healthz`
2. **Test API:** Visit `https://your-service.onrender.com/api/v1/inference/models`
3. **Update Frontend:** Configure your frontend's `api.ts` with the new backend URL
4. **Monitor Logs:** Check Render dashboard for any startup errors

## Important Notes

- **Free Tier Limitations:**
  - Service spins down after 15 minutes of inactivity
  - First request after spin-down takes 30-60 seconds (cold start)
  - 750 hours/month free compute
  
- **Cold Start with TensorFlow:**
  - Model loading on cold start may take extra time
  - Consider keeping service warm with periodic health checks
  
- **CORS Configuration:**
  - Update `XRAY_CORS_ORIGINS` environment variable with your frontend domain
  - Remove `localhost` origins in production

## Troubleshooting

### Build Fails - Out of Memory
- Reduce dependencies or use smaller TensorFlow build
- Upgrade to paid tier with more resources

### Models Not Found
- Verify `XRAY_MODEL_DIR` path
- Check model files are accessible in the deployment
- Review build logs for download errors

### Service Won't Start
- Check logs in Render dashboard
- Verify start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Ensure all dependencies are installed

### Slow Response Times
- TensorFlow model inference is CPU-intensive
- Consider upgrading to higher instance type
- Implement response caching if appropriate
