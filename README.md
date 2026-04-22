# Render deployment bundle for ESA decision support

This bundle contains the minimum files required to deploy the local ESA decision-support prototype to Render.

## Included
- `webapp/app.py`: Streamlit UI
- `webapp/inference.py`: model and phenotype inference logic
- `clinical_prediction/best_model_*.joblib`: trained prediction model
- `phenotype_clustering/cluster_metadata.json`: phenotype centroid and naming metadata
- `requirements.txt`: Python dependencies
- `.python-version`: pins a Render-safe Python runtime
- `render.yaml`: Render service definition

## Deploy on Render
1. Push the full contents of this `render_deploy/` directory to the root of a GitHub repository.
2. In Render, create a new Blueprint or Web Service from that repository.
3. If you use Blueprint, Render reads `render.yaml` automatically.
4. If you use a manual Web Service setup:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run webapp/app.py --server.port $PORT --server.address 0.0.0.0`
5. Wait for the deploy to finish, then open the generated `*.onrender.com` URL.

## If `render_deploy` is a subfolder in your repo
- Set the service Root Directory to `render_deploy`, or
- move this folder's contents to the repository root.

## Important note
This deployment bundle intentionally excludes the raw CSV source data. It only contains the files required for online inference.
