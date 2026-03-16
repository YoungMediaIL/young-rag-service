# Young Media RAG Service

Multi-client AI knowledge base powered by Vertex AI RAG Engine + Gemini 2.0 Flash.

Each client gets a dedicated corpus synced with their Google Drive folder. Query the knowledge base via API from n8n, Zite, or any HTTP client.

## Architecture

```
Airtable (docs table)          Google Drive (client folders)
       ↓                              ↓
    n8n Trigger                  Vertex AI RAG Engine
  (new/updated doc)              (import_files from Drive)
       ↓                              ↓
  POST /upload-url              POST /import-drive
       ↓                              ↓
              Corpus per Client
                    ↓
              POST /query
              (Gemini RAG Tool)
                    ↓
            Zite / n8n / API
```

## Quick Start

### 1. Deploy to Cloud Run

```bash
chmod +x deploy.sh
./deploy.sh YOUR_PROJECT_ID
```

### 2. Grant Drive Access

Find the service account:
```bash
# Project number: 99408149559
# Service account: service-99408149559@gcp-sa-vertex-rag.iam.gserviceaccount.com
```

Share client Drive folders with this email (Viewer permission).

### 3. Create Client Corpus

```bash
curl -X POST https://SERVICE_URL/corpus \
  -H "Content-Type: application/json" \
  -d '{"display_name": "פרופיטזון", "description": "Knowledge base for Profitzone"}'
```

### 4. Import Drive Folder

```bash
curl -X POST https://SERVICE_URL/import-drive \
  -H "Content-Type: application/json" \
  -d '{"corpus_id": "CORPUS_ID", "folder_id": "DRIVE_FOLDER_ID"}'
```

### 5. Query

```bash
curl -X POST https://SERVICE_URL/query \
  -H "Content-Type: application/json" \
  -d '{"query": "איך עובדת האוטומציה של פרופיטזון?", "corpus_id": "CORPUS_ID"}'
```

## API Reference

### Health & Info

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health check |

### Corpus Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/corpus` | POST | Create corpus for a client |
| `/corpus` | GET | List all corpora |
| `/corpus/{id}` | GET | Get corpus details |
| `/corpus/{id}` | DELETE | Delete corpus |

### Google Drive Integration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/import-drive` | POST | Import Drive folder to corpus |
| `/sync-drive` | POST | Re-sync Drive folder (only changed files) |

### Document Upload

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload text content |
| `/upload-url` | POST | Upload from Drive URL or GCS |
| `/upload-file` | POST | Upload file (PDF, DOCX, etc.) |

### Query

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Query with Gemini RAG Tool |
| `/query-multi` | POST | Query across multiple corpora |
| `/retrieve` | POST | Retrieve chunks only (debug) |

### Document Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/documents/{corpus_id}` | GET | List documents in corpus |
| `/documents/{corpus_id}/{file_id}` | GET | Get document details |
| `/documents/{corpus_id}/{file_id}` | DELETE | Delete document |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_ID` | young-media-ltd-1669631043907 | GCP Project ID |
| `LOCATION` | europe-west1 | Vertex AI region |
| `GCS_BUCKET` | {PROJECT_ID}-rag-uploads | GCS bucket for temp uploads |
| `GEMINI_MODEL` | gemini-2.0-flash | Model for query responses |
| `CHUNK_SIZE` | 512 | Chunk size in tokens |
| `CHUNK_OVERLAP` | 128 | Chunk overlap in tokens |

## n8n Integration Examples

### Create Corpus for New Client
```
HTTP Request:
  Method: POST
  URL: https://SERVICE_URL/corpus
  Body: { "display_name": "{{ $json.client_name }}" }
```

### Sync Drive Folder (Daily)
```
Schedule Trigger: Every day at 02:00
HTTP Request:
  Method: POST
  URL: https://SERVICE_URL/sync-drive
  Body: {
    "corpus_id": "{{ $json.corpus_id }}",
    "folder_id": "{{ $json.drive_folder_id }}"
  }
```

### Upload Doc from Airtable
```
Airtable Trigger: Record updated in docs table
HTTP Request:
  Method: POST
  URL: https://SERVICE_URL/upload-url
  Body: {
    "corpus_id": "{{ $json.corpus_id }}",
    "url": "{{ $json.documentation_url }}"
  }
```

### Query
```
HTTP Request:
  Method: POST
  URL: https://SERVICE_URL/query
  Body: {
    "query": "{{ $json.user_question }}",
    "corpus_id": "{{ $json.corpus_id }}"
  }
```

## Airtable Schema

### Clients table — new fields:
- `Corpus ID` — Vertex AI RAG Corpus ID
- `Drive Folder ID` — Google Drive folder ID
- `Drive Folder URL` — Google Drive folder URL

### docs table — new fields:
- `RAG File ID` — Vertex AI RAG File ID
- `RAG Sync Status` — Last sync status (synced/error)
- `Last Synced` — Timestamp of last sync

## Key Features vs YAEL v1

| Feature | YAEL v1 | Young Media RAG |
|---------|---------|-----------------|
| Drive import | Via GCS only | Direct Drive import ✅ |
| Auto-sync | Manual | Re-import skips unchanged files ✅ |
| Query mode | Manual prompt | Gemini RAG Tool ✅ |
| Multi-corpus query | No | Yes ✅ |
| Drive URL detection | No | Auto-extracts file/folder ID ✅ |

## Cost Estimation

- Cloud Run: ~$0-2/month (scales to zero)
- Vertex AI RAG Engine: Minimal per query
- Embedding: ~$0.0001 per 1K characters
- Gemini 2.0 Flash: ~$0.075 per 1M input tokens

## Troubleshooting

### "Permission denied" on Drive import
Grant Viewer to `service-{NUMBER}@gcp-sa-vertex-rag.iam.gserviceaccount.com` on the Drive folder.

### "Corpus not found"
Verify corpus_id. Use `GET /corpus` to list all.

### Files not syncing
Run `/sync-drive` — unchanged files are skipped automatically. Check if the file was actually modified.

---

*Built by Yair Hefetz | Young Media Automations | March 2026*
