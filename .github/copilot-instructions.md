# Flipped Classroom RAG + ML Platform — AI Workspace Instructions

**Project:** Flipped Classroom learning platform with RAG-powered Q&A and ML-based performance prediction for CSE education.

## Quick Start

### Setup & Local Development
```bash
cd flipped_classroom_project
python -m venv ../.venv
../.venv/Scripts/activate  # Windows PowerShell
pip install -r requirements.txt

# PyTorch (CPU): pre-installed; if needed, install explicitly:
# pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

python manage.py migrate
python manage.py create_admin  # Creates admin user if missing
python manage.py runserver 8000
```

**Default Admin:** Username `admin`, password `admin` (from `create_admin` command; change in production).

### Tech Stack
| Component | Stack |
|-----------|-------|
| **Backend** | Django 4.2.7, Python 3.11 |
| **Database** | SQLite (dev), PostgreSQL (production via Railway/Render) |
| **Frontend** | HTML5, CSS3, Bootstrap 5, Vanilla JS |
| **Storage** | Cloudinary (videos, materials, profile pics) |
| **RAG/AI** | Groq LLM (llama-3.1-8b-instant), FAISS (search), sentence-transformers (embeddings) |
| **ML** | Scikit-learn (regression/classification), joblib (model persistence) |

## Architecture & Key Components

### Core Application Structure
```
flipped_classroom_project/
├── flipped_app/               # Main Django app
│   ├── models.py              # Subject, StudentProfile, VideoLecture, Quiz, etc.
│   ├── views.py               # Auth, dashboard, quiz, assignment, RAG endpoints
│   ├── signals.py             # Auto-create StudentProfile on user registration
│   ├── forms.py               # ModelForms for user input
│   ├── urls.py                # Routing for 30+ endpoints
│   └── tests/
│       └── test_regressions.py    # Access control, due-date validation
│
├── rag_engine/                # RAG System (Q&A over knowledge base)
│   ├── chat.py                # Groq LLM integration, streaming support
│   ├── retriever.py           # FAISS index search, context retrieval
│   ├── indexer.py             # Build/rebuild search index from documents
│   ├── embedding_model.py     # sentence-transformers model loader
│   └── saved_index/
│       └── index.faiss        # Persisted FAISS vector index
│
├── ml_model/                  # ML Training & Prediction
│   ├── model_training.py      # Regression (Linear, RF) & Classification (LR, DT, RF)
│   ├── prediction.py          # Load models and predict student performance
│   ├── dataset.csv            # Training data (student metrics → performance label)
│   └── saved_models/          # joblib-serialized sklearn models
│
├── rag_knowledge/             # Knowledge base (indexed by RAG)
│   ├── AIML.txt, CN.txt, DS.txt, etc.
│   └── Used by indexer.py to populate FAISS
│
├── manage.py                  # Django CLI
├── requirements.txt           # All dependencies (see below)
└── render.yaml                # Production deployment on Render/Railway
```

### Key Models & Database Schema
- **User & Profiles:** `User` (Django auth) → `StudentProfile` / `TeacherProfile` (OneToOne)
- **Learning Content:** `Subject`, `VideoLecture`, `StudyMaterial`
- **Assessments:** `Quiz`, `QuizQuestion`, `Assignment`, `AssignmentSubmission`
- **Engagement & Data:** `VideoWatchHistory`, `QuizAttempt`, `Attendance`, `StudentPerformance` (performance label & ML prediction)
- **Communication:** `ChatMessage` (RAG Q&A history)

### Critical Configuration

**Environment Variables (`.env` or Render/Railway console):**
```
DEBUG=True                    # False in production
SECRET_KEY=...                # Auto-generated on Render
GROQ_API_KEY=...              # Required for RAG chat
CLOUDINARY_URL=...            # Required for media persistence
DATABASE_URL=...              # PostgreSQL URL (production)
ALLOWED_HOSTS=localhost,127.0.0.1  (dev) or .onrender.com,.up.railway.app (prod)
RAG_ENABLE_WEB_SEARCH=false   # Optional: enable external web search in RAG
```

**Key Settings (`settings.py`):**
- Uses **.env file** for secrets (not commited to git)
- **SECURE_PROXY_SSL_HEADER** set for Railway/Render HTTPS termination
- **Cloudinary Storage** for `MEDIA_ROOT` (videos persist across deploys)
- **CSRF_TRUSTED_ORIGINS** auto-built from `ALLOWED_HOSTS` or explicit env var

## Workflows & Common Tasks

### 1. Adding a New Quiz/Assignment
1. **Login as Teacher** → Go to Dashboard → Create Quiz/Assignment
2. Create questions → Set due date → Publish
3. Students see in their dashboard, auto-tracked in `QuizAttempt` / `AssignmentSubmission`
4. **Before moving to production:** Export submissions to CSV via Admin panel or Django ORM

### 2. ML Model Training & Predictions
```bash
# In ml_model/:
python generate_dataset.py    # Collect student engagement metrics → CSV
python model_training.py      # Train regression & classification models → saved_models/
# Models auto-load on student performance page (views.py: predict_performance())
```
**Note:** Models are trained offline; predictions run in `views.py` on demand.

### 3. RAG Knowledge Base Updates
1. Add `.txt` files to `rag_knowledge/` or upload via Teacher Dashboard
2. **Trigger rebuild:** POST `/rebuild-rag/` (admin-only endpoint)
3. Rebuilds FAISS index from all `.txt` files in `rag_knowledge/`
4. **Students** can now query updated knowledge via `/chat/` endpoint (SSE streaming)

### 4. Testing
```bash
python manage.py test flipped_app.tests
# Runs test_regressions.py: access control, due-date blocking, form validation
```
**Convention:** Use `@override_settings(DEBUG=True)` and `@patch` for external API mocks.

### 5. Production Deployment
- **Render / Railway:** Auto-deploys from GitHub on push  
  - Uses `render.yaml` / `railway.json` configuration
  - Runs migrations (`python manage.py migrate --noinput`)
  - Collects static files, installs CPU PyTorch  
  - **First deploy only:** POST admin creation endpoint or manually create superuser
- **Database:** Auto-provisioned PostgreSQL on Render/Railway
- **Media Storage:** Cloudinary (set `CLOUDINARY_URL` env var)

---

## Key Patterns & Conventions

### User Role Detection
```python
# views.py patterns:
def is_teacher(user):
    return hasattr(user, 'teacher_profile') or user.is_staff

def is_student(user):
    return hasattr(user, 'student_profile')
```

### Access Control Pattern
- Use **`@login_required`** for authenticated endpoints
- Use **`@user_passes_test(is_student)`** or **`@user_passes_test(is_teacher)`** for role-based views
- See `test_regressions.py` for examples: blocks past-due quizzes, prevents unauthorized submissions

### RAG Integration Pattern
```python
# views.py: chat_view()
from rag_engine.chat import ask_rag
from rag_engine.retriever import get_context

context = get_context(user_query, top_k=3)
response = ask_rag(query, context, stream=True)  # Returns generator for SSE
```

### ML Integration Pattern
```python
# views.py: predict_performance()
from ml_model.prediction import predict_performance_class, predict_performance_final_score
pred = predict_performance_class(student)  # Returns label + confidence
```

### Media Storage Pattern
- **User Uploads:** Saved to Cloudinary via `django-cloudinary-storage`
- **Model Field:** `models.FileField()` or `models.ImageField()` auto-routes through Cloudinary
- **Access:** URL auto-generated, includes Cloudinary CDN for fast retrieval

### Signal-Based Auto-Enrollment
```python
# signals.py: On new User creation, auto-create StudentProfile
# Teacher/Admin: Created via Django admin (creates TeacherProfile manually)
```

---

## Common Debugging & Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **`GROQ_API_KEY` not set** | Missing env var | Set in `.env` or Render/Railway console before running RAG chat |
| **Cloudinary uploads fail** | `CLOUDINARY_URL` not set | Add Cloudinary URL env var; check credentials on Cloudinary dashboard |
| **FAISS index stale** | Knowledge base updated but index not rebuilt | POST `/rebuild-rag/` (admin-only) to rebuild index |
| **Django migration errors** | New fields in models | Run `python manage.py makemigrations && migrate` |
| **Tests pass locally, fail on Render** | env vars not set | Ensure all `GROQ_API_KEY`, `CLOUDINARY_URL`, `SECRET_KEY` are in Render/Railway console |
| **ML model not found** | Model training not run | Run `python ml_model/model_training.py` to generate models in `saved_models/` |
| **Past-due quiz not blocked** | Bug in due-date check | See `test_take_quiz_blocks_past_due_date` test for expected behavior |

---

## AI Assistant Notes

### What to Do Before Generating Code
1. **Ask about env setup:** Is `.env` file in place? Are Groq/Cloudinary keys available?
2. **Check deployed vs. local:** Is this a local dev fix or production bug? Affects logging/error handling.
3. **Verify migrations:** After model changes, always generate and test migrations locally first.

### What NOT to Do
- **Don't commit secrets** (API keys, SECRET_KEY) to git; use `.env` or platform env vars
- **Don't modify `render.yaml` carelessly** — it controls production deployments
- **Don't train ML models in views** — do offline in management commands or batch jobs
- **Don't hardcode Cloudinary URLs** — use Django's FileField abstraction

### Testing Philosophy
- **Unit test:** Model logic, form validation, helper functions
- **Integration test:** Auth flows, due-date blocking, RAG index rebuilds (use mocks for external APIs)
- **No API mocking needed for:** Database queries, Django ORM, file operations

---

## Useful Commands

| Command | Purpose |
|---------|---------|
| `python manage.py shell` | Interactive Django shell (test queries, debug models) |
| `python manage.py dumpdata > backup.json` | Export all data |
| `python manage.py loaddata backup.json` | Restore from backup |
| `python manage.py createsuperuser` | Create admin user (if auto-create fails) |
| `python ml_model/generate_dataset.py` | Collect engagement metrics → CSV |
| `python ml_model/model_training.py` | Train & save ML models |
| `curl -X POST http://localhost:8000/rebuild-rag/` | Rebuild RAG index (requires admin token) |

---

## Related Documentation

- [Django 4.2 Docs](https://docs.djangoproject.com/en/4.2/) — Models, views, forms, testing
- [FAISS Docs](https://github.com/facebookresearch/faiss) — Vector search indexing
- [Groq API Docs](https://console.groq.com/docs) — LLM API reference
- [Scikit-learn](https://scikit-learn.org/) — ML models for prediction
- [Render Blueprint Spec](https://render.com/docs/blueprint-spec) — Production deployment config
