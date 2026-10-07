import os
from django.core.wsgi import get_wsgi_application

# Ensure PyTorch compatibility patches are applied before any models load
try:
    from flipped_app.torch_patch import patch_torch_compat
    patch_torch_compat()
except Exception:
    pass

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flipped_classroom_project.settings')
application = get_wsgi_application()
