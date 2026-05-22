"""
Django settings for flipped_classroom_project.
"""

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var safely."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}

# Load .env file if present (keeps secrets out of source control)
_env_file = BASE_DIR / '.env'
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip())

DEBUG = _env_bool('DEBUG', default=False)

SECRET_KEY = os.environ.get('SECRET_KEY', '').strip()
if not SECRET_KEY:
    if DEBUG:
        # Local-only fallback so developers can run without extra setup.
        SECRET_KEY = 'dev-only-secret-key-change-before-deploy'
    else:
        raise RuntimeError('SECRET_KEY is required when DEBUG=False')

# In production set ALLOWED_HOSTS=.up.railway.app in env vars
# Converts *.domain to .domain (Django uses leading-dot for subdomain wildcard)
_default_hosts = 'localhost,127.0.0.1'
_raw_hosts = os.environ.get('ALLOWED_HOSTS', _default_hosts)
ALLOWED_HOSTS = [
    h.strip()[1:] if h.strip().startswith('*.') else h.strip()
    for h in _raw_hosts.split(',') if h.strip()
]

# ── Production Security Settings (auto-enabled when DEBUG=False) ──────────
if not DEBUG:
    # Railway (and most PaaS) terminate SSL at the proxy; the app receives plain HTTP.
    # Tell Django to trust the X-Forwarded-Proto header so it knows the original
    # request was HTTPS and won't keep redirecting.
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    # Do NOT set SECURE_SSL_REDIRECT=True — Railway handles HTTPS at the edge.
    SECURE_SSL_REDIRECT = False
    SECURE_HSTS_SECONDS = 31536000          # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    # Secure cookies
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    # Trusted origins for CSRF when behind a proxy / HTTPS.
    # Build from env var, or fall back to https:// prefix of every ALLOWED_HOST.
    _csrf_origins = os.environ.get('CSRF_TRUSTED_ORIGINS', '')
    if _csrf_origins:
        CSRF_TRUSTED_ORIGINS = [o.strip() for o in _csrf_origins.split(',') if o.strip()]
    else:
        # Auto-build from ALLOWED_HOSTS so POST requests work without manual config
        CSRF_TRUSTED_ORIGINS = [
            f'https://{h}' for h in ALLOWED_HOSTS
            if h not in ('*', 'localhost', '127.0.0.1')
        ] or ['https://*.up.railway.app', 'https://*.onrender.com']
    # Prevent browsers from sniffing MIME types
    SECURE_CONTENT_TYPE_NOSNIFF = True
    # Note: SECURE_BROWSER_XSS_FILTER was removed in Django 4.0 (deprecated header,
    # modern browsers ignore it; use Content-Security-Policy instead).

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'cloudinary',
    'cloudinary_storage',
    'crispy_forms',
    'crispy_bootstrap5',
    'flipped_app',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'flipped_classroom_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'flipped_app' / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'flipped_classroom_project.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Use PostgreSQL in production when DATABASE_URL env var is set (Railway / Render)
_db_url = os.environ.get('DATABASE_URL')
if _db_url:
    import dj_database_url
    DATABASES['default'] = dj_database_url.config(
        default=_db_url,
        conn_max_age=600,
        ssl_require=not DEBUG,
    )
    # Add connection retry logic for Render cold starts
    # When Render deploys, the DB may not be immediately available
    DATABASES['default']['CONN_MAX_AGE'] = 600
    DATABASES['default']['OPTIONS'] = {
        'connect_timeout': 10,
        'options': '-c statement_timeout=15000',  # 15s statement timeout
        **({"sslmode": "require"} if not DEBUG else {}),
    }

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
# Keep empty to avoid duplicate collection of flipped_app/static.
# AppDirectoriesFinder already discovers app static files (including admin assets).
STATICFILES_DIRS = []
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_FINDERS = [
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
]
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
}

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# ── Cloudinary (persistent media storage for production) ──────────────
# Set CLOUDINARY_URL in environment for production (Render/Railway).
# Format: cloudinary://api_key:api_secret@cloud_name
# Locally, falls back to local FileSystemStorage when CLOUDINARY_URL is not set.
_cloudinary_url = os.environ.get('CLOUDINARY_URL', '')
if _cloudinary_url:
    import cloudinary
    _parts = _cloudinary_url.replace('cloudinary://', '').split('@')
    _creds, _cloud = _parts[0], _parts[1]
    _api_key, _api_secret = _creds.split(':')
    cloudinary.config(
        cloud_name=_cloud,
        api_key=_api_key,
        api_secret=_api_secret,
        secure=True,
    )
    STORAGES['default'] = {'BACKEND': 'cloudinary_storage.storage.MediaCloudinaryStorage'}

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

CRISPY_ALLOWED_TEMPLATE_PACKS = 'bootstrap5'
CRISPY_TEMPLATE_PACK = 'bootstrap5'

LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/dashboard/'
LOGOUT_REDIRECT_URL = '/login/'

# ── RAG Chatbot Settings ──────────────────────────────────
# API key is loaded from .env file (see BASE_DIR/.env) — never hardcode here
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
RAG_ENABLE_WEB_SEARCH = _env_bool('RAG_ENABLE_WEB_SEARCH', default=False)
RAG_INDEX_PATH = BASE_DIR / 'rag_engine' / 'saved_index'
RAG_KNOWLEDGE_PATH = BASE_DIR / 'rag_knowledge'

# ── Suppress noisy but harmless ML library warnings ──────
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
import warnings
warnings.filterwarnings('ignore', message='.*position_ids.*')
warnings.filterwarnings('ignore', message='.*HF Hub.*')

# ── Session / Auto-Logout Settings ───────────────────────
# Expire session after 2 hours of inactivity
SESSION_COOKIE_AGE = 7200          # 2 hours in seconds
SESSION_SAVE_EVERY_REQUEST = True  # Reset the 2-hr timer on every request
SESSION_EXPIRE_AT_BROWSER_CLOSE = True  # Also logout when browser is closed

# ── Logging: real-time dataset events visible in console ─
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'fliplearn': {
            'format': '[{asctime}] {levelname} {name} — {message}',
            'style': '{',
            'datefmt': '%H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'fliplearn',
        },
    },
    'loggers': {
        'fliplearn.realtime_dataset': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'fliplearn.signals': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'django.request': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
    },
}
