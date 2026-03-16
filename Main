"""
Young Media RAG Service
Cloud Run service for multi-client knowledge base using Vertex AI RAG Engine

Based on YAEL RAG architecture, customized for Young Media.
Each client gets a dedicated corpus with Drive folder sync.
"""

import os
import json
import logging
from typing import Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import vertexai
from vertexai.preview import rag
from vertexai.preview.generative_models import GenerativeModel, Tool
from google.cloud import storage
import tempfile
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== Configuration ==============

PROJECT_ID = os.environ.get("PROJECT_ID", "young-media-ltd-1669631043907")
LOCATION = os.environ.get("LOCATION", "europe-west1")
DEFAULT_CORPUS_ID = os.environ.get("CORPUS_ID", "")  # Optional: default corpus
GCS_BUCKET = os.environ.get("GCS_BUCKET", f"{PROJECT_ID}-rag-uploads")

# Embedding model for Hebrew + English support
EMBEDDING_MODEL = "publishers/google/models/text-multilingual-embedding-002"

# Gemini model for query responses
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Chunking configuration
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "128"))

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

app = FastAPI(
    title="Young Media RAG Service",
    description="Multi-client knowledge base powered by Vertex AI RAG Engine",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Models ==============

class CreateCorpusRequest(BaseModel):
    """Create a new corpus for a client"""
    display_name: str
    description: Optional[str] = None

class ImportDriveRequest(BaseModel):
    """Import files from a Google Drive folder"""
    corpus_id: str
    folder_id: str
    resource_type: str = "RESOURCE_TYPE_FOLDER"  # or RESOURCE_TYPE_FILE

class SyncDriveRequest(BaseModel):
    """Sync (re-import) a Drive folder — only changed files get reindexed"""
    corpus_id: str
    folder_id: str

class UploadTextRequest(BaseModel):
    """Upload text content directly"""
    corpus_id: str
    content: str
    display_name: str
    description: Optional[str] = None

class UploadUrlRequest(BaseModel):
    """Upload from a GCS or Google Drive URL"""
    corpus_id: str
    url: str
    display_name: Optional[str] = None

class QueryRequest(BaseModel):
    """Query the knowledge base"""
    query: str
    corpus_id: str
    similarity_top_k: int = 10
    vector_distance_threshold: float = 0.5
    use_rag_tool: bool = True  # Use Gemini RAG Tool integration

class MultiCorpusQueryRequest(BaseModel):
    """Query across multiple corpora"""
    query: str
    corpus_ids: List[str]
    similarity_top_k: int = 10
    vector_distance_threshold: float = 0.5


# ============== Helper Functions ==============

def get_corpus_name(corpus_id: str) -> str:
    """Build full corpus resource name"""
    return f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{corpus_id}"

def extract_file_id(file_name: str) -> str:
    """Extract file ID from full resource name"""
    if "/ragFiles/" in file_name:
        return file_name.split("/ragFiles/")[-1]
    return file_name

def extract_corpus_id(corpus_name: str) -> str:
    """Extract corpus ID from full resource name"""
    if "/ragCorpora/" in corpus_name:
        return corpus_name.split("/ragCorpora/")[-1]
    return corpus_name


# ============== Health & Info ==============

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy",
        "service": "Young Media RAG Service",
        "version": "1.0.0",
        "project": PROJECT_ID,
        "location": LOCATION,
        "embedding_model": EMBEDDING_MODEL,
        "gemini_model": GEMINI_MODEL
    }

@app.get("/health")
async def health():
    """Detailed health check — tests Vertex AI connectivity"""
    try:
        corpora = list(rag.list_corpora())
        return {
            "status": "healthy",
            "corpus_count": len(corpora),
            "project": PROJECT_ID,
            "location": LOCATION
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


# ============== Corpus Management ==============

@app.post("/corpus")
async def create_corpus(request: CreateCorpusRequest):
    """
    Create a new RAG corpus for a client.
    Each client should have their own corpus.
    Returns corpus_id to store in Airtable.
    """
    try:
        logger.info(f"Creating corpus: {request.display_name}")

        embedding_model_config = rag.EmbeddingModelConfig(
            publisher_model=EMBEDDING_MODEL
        )

        corpus = rag.create_corpus(
            display_name=request.display_name,
            description=request.description or f"Knowledge base for {request.display_name}",
            embedding_model_config=embedding_model_config,
        )

        corpus_id = extract_corpus_id(corpus.name)
        logger.info(f"Created corpus: {corpus_id}")

        return {
            "success": True,
            "corpus_id": corpus_id,
            "corpus_name": corpus.name,
            "display_name": corpus.display_name
        }

    except Exception as e:
        logger.error(f"Create corpus failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/corpus")
async def list_corpora():
    """List all corpora with document counts"""
    try:
        corpora = list(rag.list_corpora())
        result = []

        for c in corpora:
            corpus_id = extract_corpus_id(c.name)
            try:
                files = list(rag.list_files(corpus_name=c.name))
                doc_count = len(files)
            except Exception:
                doc_count = 0

            result.append({
                "corpus_id": corpus_id,
                "display_name": c.display_name if hasattr(c, 'display_name') else None,
                "description": c.description if hasattr(c, 'description') else None,
                "document_count": doc_count
            })

        result.sort(key=lambda x: x["document_count"], reverse=True)
        return {"total": len(result), "corpora": result}

    except Exception as e:
        logger.error(f"List corpora failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/corpus/{corpus_id}")
async def get_corpus(corpus_id: str):
    """Get corpus details"""
    try:
        corpus_name = get_corpus_name(corpus_id)
        corpus = rag.get_corpus(name=corpus_name)
        return {
            "corpus_id": corpus_id,
            "display_name": corpus.display_name,
            "description": corpus.description if hasattr(corpus, 'description') else None
        }
    except Exception as e:
        logger.error(f"Get corpus failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/corpus/{corpus_id}")
async def delete_corpus(corpus_id: str):
    """Delete a corpus and all its documents"""
    try:
        corpus_name = get_corpus_name(corpus_id)
        rag.delete_corpus(name=corpus_name)
        return {"success": True, "message": f"Corpus {corpus_id} deleted"}
    except Exception as e:
        logger.error(f"Delete corpus failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Google Drive Integration ==============

@app.post("/import-drive")
async def import_from_drive(request: ImportDriveRequest):
    """
    Import files directly from a Google Drive folder into a corpus.
    
    Prerequisites:
    - Grant "Viewer" permission to the Vertex RAG Data Service Agent:
      service-{PROJECT_NUMBER}@gcp-sa-vertex-rag.iam.gserviceaccount.com
      on the Drive folder.
    
    The folder_id is the ID from the Google Drive URL:
    https://drive.google.com/drive/folders/{folder_id}
    """
    try:
        corpus_name = get_corpus_name(request.corpus_id)
        logger.info(f"Importing Drive folder {request.folder_id} to corpus {request.corpus_id}")

        response = rag.import_files(
            corpus_name=corpus_name,
            paths=[],  # empty — using google_drive_source instead
            google_drive_source=rag.GoogleDriveSource(
                resource_ids=[
                    rag.GoogleDriveSource.ResourceId(
                        resource_id=request.folder_id,
                        resource_type=request.resource_type,
                    )
                ]
            ),
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        logger.info(f"Drive import response: {response}")

        # Count files after import
        files = list(rag.list_files(corpus_name=corpus_name))

        return {
            "success": True,
            "message": f"Drive folder imported to corpus {request.corpus_id}",
            "corpus_id": request.corpus_id,
            "folder_id": request.folder_id,
            "total_files_in_corpus": len(files),
            "import_result": str(response)
        }

    except Exception as e:
        logger.error(f"Drive import failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync-drive")
async def sync_drive(request: SyncDriveRequest):
    """
    Re-import a Drive folder. Vertex AI automatically:
    - Skips files that haven't changed
    - Re-indexes files that were modified
    - Adds new files
    
    Call this periodically (e.g., daily via n8n) to keep knowledge base current.
    """
    try:
        corpus_name = get_corpus_name(request.corpus_id)
        logger.info(f"Syncing Drive folder {request.folder_id} for corpus {request.corpus_id}")

        # Count files before sync
        files_before = list(rag.list_files(corpus_name=corpus_name))

        response = rag.import_files(
            corpus_name=corpus_name,
            paths=[],
            google_drive_source=rag.GoogleDriveSource(
                resource_ids=[
                    rag.GoogleDriveSource.ResourceId(
                        resource_id=request.folder_id,
                        resource_type="RESOURCE_TYPE_FOLDER",
                    )
                ]
            ),
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # Count files after sync
        files_after = list(rag.list_files(corpus_name=corpus_name))

        return {
            "success": True,
            "message": "Drive sync completed",
            "corpus_id": request.corpus_id,
            "folder_id": request.folder_id,
            "files_before": len(files_before),
            "files_after": len(files_after),
            "new_files": len(files_after) - len(files_before),
            "sync_time": datetime.utcnow().isoformat(),
            "import_result": str(response)
        }

    except Exception as e:
        logger.error(f"Drive sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Document Upload ==============

@app.post("/upload")
async def upload_text(request: UploadTextRequest):
    """
    Upload text content to a corpus.
    Content is first uploaded to GCS, then imported to Vertex AI.
    """
    try:
        corpus_name = get_corpus_name(request.corpus_id)
        logger.info(f"Uploading text: {request.display_name} to corpus {request.corpus_id}")

        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}_{request.display_name}"
        if not filename.endswith('.txt'):
            filename += '.txt'

        # Upload to GCS
        gcs_uri = _upload_to_gcs(request.content, filename)

        try:
            response = rag.import_files(
                corpus_name=corpus_name,
                paths=[gcs_uri],
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

            # Find uploaded file
            uploaded_file = _find_uploaded_file(corpus_name, unique_id)
            file_id = extract_file_id(uploaded_file.name) if uploaded_file else None

            return {
                "success": True,
                "message": f"Document '{request.display_name}' uploaded",
                "document_id": file_id,
                "corpus_id": request.corpus_id
            }

        finally:
            _delete_from_gcs(filename)

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-url")
async def upload_from_url(request: UploadUrlRequest):
    """
    Import a document from a Google Drive URL or GCS URI.
    Supports: gs://bucket/path or Google Drive file/folder IDs
    """
    try:
        corpus_name = get_corpus_name(request.corpus_id)
        logger.info(f"Uploading from URL: {request.url} to corpus {request.corpus_id}")

        # Detect if it's a Google Drive URL and extract ID
        drive_id = _extract_drive_id(request.url)

        if drive_id:
            # Import from Drive directly
            response = rag.import_files(
                corpus_name=corpus_name,
                paths=[],
                google_drive_source=rag.GoogleDriveSource(
                    resource_ids=[
                        rag.GoogleDriveSource.ResourceId(
                            resource_id=drive_id,
                            resource_type="RESOURCE_TYPE_FILE",
                        )
                    ]
                ),
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        else:
            # Assume GCS URI
            response = rag.import_files(
                corpus_name=corpus_name,
                paths=[request.url],
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

        # Find the most recent file
        uploaded_file = _find_uploaded_file(corpus_name, "")
        file_id = extract_file_id(uploaded_file.name) if uploaded_file else None

        return {
            "success": True,
            "message": "Document uploaded from URL",
            "document_id": file_id,
            "corpus_id": request.corpus_id,
            "source_url": request.url
        }

    except Exception as e:
        logger.error(f"URL upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    corpus_id: str = Form(...),
    display_name: str = Form(None),
    description: str = Form(None)
):
    """Upload a file directly (PDF, DOCX, TXT, MD, HTML)"""
    try:
        corpus_name = get_corpus_name(corpus_id)
        doc_name = display_name or file.filename
        logger.info(f"Uploading file: {doc_name} to corpus {corpus_id}")

        unique_id = str(uuid.uuid4())[:8]
        suffix = os.path.splitext(file.filename)[1] or '.txt'
        unique_filename = f"{unique_id}_{doc_name}{suffix}"

        content = await file.read()

        # Upload to GCS
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(f"uploads/{unique_filename}")
        blob.upload_from_string(content)
        gcs_uri = f"gs://{GCS_BUCKET}/uploads/{unique_filename}"

        try:
            response = rag.import_files(
                corpus_name=corpus_name,
                paths=[gcs_uri],
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

            uploaded_file = _find_uploaded_file(corpus_name, unique_id)
            file_id = extract_file_id(uploaded_file.name) if uploaded_file else None

            return {
                "success": True,
                "message": f"File '{doc_name}' uploaded",
                "document_id": file_id,
                "corpus_id": corpus_id
            }

        finally:
            _delete_from_gcs(unique_filename)

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Query ==============

@app.post("/query")
async def query_rag(request: QueryRequest):
    """
    Query the knowledge base with Gemini RAG Tool integration.
    
    Two modes:
    - use_rag_tool=True (default): Uses Gemini's native RAG Tool for grounded responses
    - use_rag_tool=False: Manual retrieval + generation (like YAEL v1)
    """
    try:
        corpus_name = get_corpus_name(request.corpus_id)
        logger.info(f"Query: {request.query} on corpus {request.corpus_id}")

        if request.use_rag_tool:
            return await _query_with_rag_tool(request, corpus_name)
        else:
            return await _query_manual(request, corpus_name)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-multi")
async def query_multi_corpus(request: MultiCorpusQueryRequest):
    """
    Query across multiple client corpora at once.
    Useful for cross-client insights or internal queries.
    """
    try:
        rag_resources = []
        for cid in request.corpus_ids:
            corpus_name = get_corpus_name(cid)
            rag_resources.append(rag.RagResource(rag_corpus=corpus_name))

        # Use RAG Tool across multiple corpora
        rag_retrieval_tool = Tool.from_retrieval(
            retrieval=rag.Retrieval(
                source=rag.VertexRagStore(
                    rag_resources=rag_resources,
                    similarity_top_k=request.similarity_top_k,
                    vector_distance_threshold=request.vector_distance_threshold,
                ),
            )
        )

        model = GenerativeModel(
            GEMINI_MODEL,
            tools=[rag_retrieval_tool],
            system_instruction=_get_system_prompt()
        )

        response = model.generate_content(request.query)

        return {
            "answer": response.text,
            "corpus_ids": request.corpus_ids,
            "query": request.query
        }

    except Exception as e:
        logger.error(f"Multi-corpus query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Document Management ==============

@app.get("/documents/{corpus_id}")
async def list_documents(corpus_id: str):
    """List all documents in a corpus"""
    try:
        corpus_name = get_corpus_name(corpus_id)
        files = rag.list_files(corpus_name=corpus_name)

        documents = []
        for f in files:
            file_id = extract_file_id(f.name)
            documents.append({
                "id": file_id,
                "name": f.name,
                "display_name": f.display_name if hasattr(f, 'display_name') else None,
                "size_bytes": f.size_bytes if hasattr(f, 'size_bytes') else None,
                "create_time": str(f.create_time) if hasattr(f, 'create_time') else None
            })

        return {
            "corpus_id": corpus_id,
            "document_count": len(documents),
            "documents": documents
        }

    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{corpus_id}/{file_id}")
async def get_document(corpus_id: str, file_id: str):
    """Get a single document by ID"""
    try:
        corpus_name = get_corpus_name(corpus_id)
        file_name = f"{corpus_name}/ragFiles/{file_id}"
        file = rag.get_file(name=file_name)

        return {
            "id": file_id,
            "name": file.name,
            "display_name": file.display_name if hasattr(file, 'display_name') else None,
            "size_bytes": file.size_bytes if hasattr(file, 'size_bytes') else None,
            "create_time": str(file.create_time) if hasattr(file, 'create_time') else None
        }

    except Exception as e:
        logger.error(f"Get document failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{corpus_id}/{file_id}")
async def delete_document(corpus_id: str, file_id: str):
    """Delete a document from a corpus"""
    try:
        corpus_name = get_corpus_name(corpus_id)
        file_name = f"{corpus_name}/ragFiles/{file_id}"
        rag.delete_file(name=file_name)
        return {"success": True, "message": f"Document {file_id} deleted"}
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Retrieval (Debug) ==============

@app.post("/retrieve")
async def retrieve_only(request: QueryRequest):
    """
    Retrieve relevant chunks without generating a response.
    Useful for debugging retrieval quality.
    """
    try:
        corpus_name = get_corpus_name(request.corpus_id)

        retrieval_response = rag.retrieval_query(
            rag_resources=[
                rag.RagResource(rag_corpus=corpus_name)
            ],
            text=request.query,
            similarity_top_k=request.similarity_top_k,
            vector_distance_threshold=request.vector_distance_threshold,
        )

        chunks = []
        if retrieval_response.contexts and retrieval_response.contexts.contexts:
            for ctx in retrieval_response.contexts.contexts:
                chunks.append({
                    "text": ctx.text,
                    "source": ctx.source_uri if hasattr(ctx, 'source_uri') else "unknown",
                    "score": ctx.distance if hasattr(ctx, 'distance') else None
                })

        return {
            "query": request.query,
            "corpus_id": request.corpus_id,
            "chunk_count": len(chunks),
            "chunks": chunks
        }

    except Exception as e:
        logger.error(f"Retrieve failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Internal Helpers ==============

async def _query_with_rag_tool(request: QueryRequest, corpus_name: str):
    """
    Query using Gemini's native RAG Tool integration.
    The model automatically retrieves relevant context and generates grounded responses.
    """
    rag_retrieval_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
                similarity_top_k=request.similarity_top_k,
                vector_distance_threshold=request.vector_distance_threshold,
            ),
        )
    )

    model = GenerativeModel(
        GEMINI_MODEL,
        tools=[rag_retrieval_tool],
        system_instruction=_get_system_prompt()
    )

    response = model.generate_content(request.query)

    return {
        "answer": response.text,
        "corpus_id": request.corpus_id,
        "query": request.query,
        "mode": "rag_tool"
    }

async def _query_manual(request: QueryRequest, corpus_name: str):
    """
    Manual query: retrieve chunks first, then generate with context.
    Provides more control over the retrieval + generation pipeline.
    """
    # Step 1: Retrieve
    retrieval_response = rag.retrieval_query(
        rag_resources=[
            rag.RagResource(rag_corpus=corpus_name)
        ],
        text=request.query,
        similarity_top_k=request.similarity_top_k,
        vector_distance_threshold=request.vector_distance_threshold,
    )

    chunks = []
    sources = []
    if retrieval_response.contexts and retrieval_response.contexts.contexts:
        for ctx in retrieval_response.contexts.contexts:
            chunk_info = {
                "text": ctx.text,
                "source": ctx.source_uri if hasattr(ctx, 'source_uri') else "unknown",
                "score": ctx.distance if hasattr(ctx, 'distance') else None
            }
            chunks.append(chunk_info)
            if chunk_info["source"] not in sources:
                sources.append(chunk_info["source"])

    if not chunks:
        return {
            "answer": "לא נמצא מידע רלוונטי בבסיס הידע.",
            "sources": [],
            "grounding_chunks": [],
            "mode": "manual"
        }

    # Step 2: Generate
    context = "\n\n---\n\n".join([c["text"] for c in chunks])
    model = GenerativeModel(GEMINI_MODEL)

    prompt = f"""{_get_system_prompt()}

מידע מבסיס הידע:
{context}

שאלה: {request.query}

תשובה:"""

    response = model.generate_content(prompt)

    return {
        "answer": response.text,
        "sources": sources,
        "grounding_chunks": chunks,
        "corpus_id": request.corpus_id,
        "query": request.query,
        "mode": "manual"
    }

def _get_system_prompt() -> str:
    """System prompt for query responses"""
    return """אתה עוזר של Young Media, סוכנות שיווק דיגיטלי.
ענה על השאלות בהתבסס על המידע בבסיס הידע.
אם המידע לא מספיק — אמור זאת בבירור.
ענה בעברית אלא אם השאלה באנגלית.
היה מדויק, ציין שמות אוטומציות, לקוחות, ופלטפורמות כשרלוונטי."""

def _extract_drive_id(url: str) -> Optional[str]:
    """Extract Google Drive file/folder ID from URL"""
    import re

    # Pattern: /d/{id} or /folders/{id}
    patterns = [
        r'/d/([a-zA-Z0-9_-]+)',
        r'/folders/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None

def _upload_to_gcs(content: str, filename: str) -> str:
    """Upload text content to GCS"""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"uploads/{filename}")
    blob.upload_from_string(content, content_type="text/plain; charset=utf-8")
    gcs_uri = f"gs://{GCS_BUCKET}/uploads/{filename}"
    logger.info(f"Uploaded to GCS: {gcs_uri}")
    return gcs_uri

def _delete_from_gcs(filename: str):
    """Delete a file from GCS"""
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(f"uploads/{filename}")
        blob.delete()
    except Exception as e:
        logger.warning(f"GCS delete failed: {e}")

def _find_uploaded_file(corpus_name: str, unique_marker: str):
    """Find uploaded file by marker or return most recent"""
    try:
        files = list(rag.list_files(corpus_name=corpus_name))
        files.sort(key=lambda f: f.create_time if hasattr(f, 'create_time') else '', reverse=True)

        if unique_marker:
            for f in files:
                if unique_marker in f.name:
                    return f

        return files[0] if files else None
    except Exception as e:
        logger.error(f"Error finding uploaded file: {e}")
        return None


# ============== Run ==============

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
