#!/bin/bash

# Young Media RAG Service — Cloud Run Deployment
# Run from Google Cloud Shell or local machine with gcloud configured

PROJECT_ID="${1:-young-media-ltd-1669631043907}"  # Young Media GCP Project
REGION="europe-west1"
SERVICE_NAME="young-rag-service"

echo "🚀 Deploying Young Media RAG Service..."
echo "   Project: $PROJECT_ID"
echo "   Region:  $REGION"
echo ""

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📦 Enabling APIs..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com

# Create GCS bucket for uploads (if not exists)
echo "🪣 Creating GCS bucket..."
gsutil mb -l $REGION gs://${PROJECT_ID}-rag-uploads 2>/dev/null || echo "Bucket already exists"

# Build and deploy
echo "🔨 Building and deploying..."
gcloud builds submit --config cloudbuild.yaml

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🔗 Service URL:"
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'

echo ""
echo "⚠️  Next steps:"
echo "   1. Vertex RAG Data Service Agent email:"
echo "      service-99408149559@gcp-sa-vertex-rag.iam.gserviceaccount.com"
echo "   2. Share client Drive folders with this service account (Viewer permission)"
echo "   3. Create corpora via POST /corpus"
echo "   4. Import Drive folders via POST /import-drive"
