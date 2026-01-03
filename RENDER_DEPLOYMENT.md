# Deploying to Render.com (Free Tier)

This guide covers deploying both the backend API and frontend to Render's free tier.

## 🚨 Critical: Model Files Storage

Your `.keras` model files (7 files, ~150-200MB each) are **NOT** in your Git repository. You **MUST** provide at least one model for the backend to work.

### Quick Solutions for Model Storage:

#### Option 1: GitHub Release (Recommended for Free Tier)

1. Create a new release in your GitHub repo
2. Upload your `.keras` files as release assets
3. Get the direct download URLs
4. Update `backend/build.sh` to download them:

```bash
curl -L -o models/ResNet50.keras "https://github.com/USERNAME/REPO/releases/download/v1.0.0/ResNet50.keras"
```

#### Option 2: Google Drive (Quick & Free)

1. Upload model files to Google Drive
2. Make them publicly accessible
3. Get direct download links using a service like https://sites.google.com/site/gdocs2direct/
4. Update `backend/build.sh` with the download commands

#### Option 3: Dropbox

1. Upload to Dropbox
2. Change `dl=0` to `dl=1` in the share link for direct download
3. Update `backend/build.sh`

## 📋 Deployment Steps

### 1. Prepare Your Repository

Make sure all files are committed:

```bash
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

### 2. Deploy to Render

#### Step 1: Sign Up

- Go to https://render.com
- Sign up with GitHub

#### Step 2: Deploy as Blueprint

1. Click **"New"** → **"Blueprint"**
2. Connect your GitHub repository
3. Render will detect `render.yaml` at the root
4. Click **"Apply"**

This will create **TWO services**:

- `xray-backend` (Web Service)
- `xray-frontend` (Static Site)

### 3. Configure Environment Variables

#### For Backend (`xray-backend`):

Go to the backend service settings → Environment:

1. **Required:** Add CORS origins after frontend deploys

   ```
   XRAY_CORS_ORIGINS=["https://xray-frontend.onrender.com"]
   ```

2. **Optional:** Add model download URL
   ```
   MODEL_DOWNLOAD_URL=https://your-storage-url.com/ResNet50.keras
   ```

#### For Frontend (`xray-frontend`):

Go to the frontend service settings → Environment:

1. **Required:** Set API base URL after backend deploys
   ```
   VITE_API_BASE=https://xray-backend.onrender.com
   ```

### 4. Trigger Rebuild

After setting environment variables:

1. Go to each service
2. Click **"Manual Deploy"** → **"Deploy latest commit"**

### 5. Test Your Deployment

#### Backend Health Check:

```
https://xray-backend.onrender.com/healthz
```

Should return: `{"status": "ok"}`

#### Available Models:

```
https://xray-backend.onrender.com/api/v1/inference/models
```

Should list available models

#### Frontend:

```
https://xray-frontend.onrender.com
```

Should load your React app

## ⚠️ Free Tier Limitations

### Backend:

- **512MB RAM** - Only ONE model loaded at a time (optimized in code)
- **Spins down after 15 min** of inactivity
- **Cold start: 30-90 seconds** - TensorFlow + model loading is slow
- **750 hours/month** free compute

### Frontend:

- **100GB bandwidth/month**
- Serves from global CDN
- No sleep/spin-down

## 🐛 Troubleshooting

### Backend Build Fails

**Error:** "No space left on device" or memory errors

- **Cause:** Model files too large or TensorFlow installation
- **Fix:** Ensure only required dependencies in requirements.txt

### Backend Returns "Model file not found"

**Error:** 500 error when making predictions

- **Cause:** Model files not downloaded during build
- **Fix:** Check `backend/build.sh` and logs, ensure download commands work

### Frontend Can't Connect to Backend

**Error:** Network errors or CORS errors in browser console

- **Cause:** Wrong `VITE_API_BASE` or missing CORS configuration
- **Fix:**
  1. Verify `VITE_API_BASE` environment variable in frontend
  2. Verify `XRAY_CORS_ORIGINS` includes frontend URL in backend
  3. Redeploy both services after changes

### Slow First Request (Cold Start)

**This is normal** on free tier:

- Service spins down after 15 minutes
- First request wakes it up (30-90 seconds)
- Consider:
  - Upgrading to paid tier
  - Using a service to ping your API every 10 minutes
  - Showing a loading message to users

### Model Loading Errors

**Error:** Out of memory when loading models

- **Cause:** 512MB limit exceeded
- **Fix:** Code already optimized to load one model at a time
- If still fails, model file may be corrupted or too large

## 🔒 Security Notes

1. **Never commit model files** to Git (already in .gitignore)
2. **Set CORS properly** - only allow your frontend domain
3. **Remove localhost** from CORS in production
4. **Use HTTPS** - Render provides this automatically

## 📊 Monitoring

### View Logs:

1. Go to service in Render dashboard
2. Click **"Logs"** tab
3. Filter by severity (Info, Warning, Error)

### Check Metrics:

- Dashboard shows CPU, memory, bandwidth usage
- Free tier: limited metrics retention

## 🚀 Next Steps

After successful deployment:

1. **Test all models** - Try each model to verify they work
2. **Monitor performance** - Check cold start times and response times
3. **Consider upgrades** if you need:
   - Faster response times
   - No cold starts
   - More memory for multiple models
   - Custom domains

## 💰 Cost Optimization

Free tier is sufficient for:

- Development/testing
- Low-traffic demos
- Portfolio projects
- Academic projects

Consider paid tier ($7-21/month) if you need:

- Production availability (no sleep)
- Faster performance
- Multiple models in memory
- Custom domain
- More bandwidth

---

## Quick Reference

### Your Service URLs (update after deployment):

- Backend: `https://xray-backend.onrender.com`
- Frontend: `https://xray-frontend.onrender.com`

### Key Files:

- [`render.yaml`](render.yaml) - Infrastructure config
- [`backend/build.sh`](backend/build.sh) - Backend build script (add model downloads here)
- [`frontend/build.sh`](frontend/build.sh) - Frontend build script
- [`backend/app/services/model_registry.py`](backend/app/services/model_registry.py) - Memory-optimized model loading

### Important Environment Variables:

Backend:

- `XRAY_CORS_ORIGINS` - Frontend URL
- `XRAY_MODEL_DIR` - Model directory path (default: /opt/render/project/src/backend/models)
- `MODEL_DOWNLOAD_URL` - (optional) URL to download models

Frontend:

- `VITE_API_BASE` - Backend API URL
