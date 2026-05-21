#!/bin/bash
# Wrapper script: wait-for-db-and-migrate.sh
# Ensures PostgreSQL is available before running Django migrations
# Used to handle Render cold starts where DB isn't immediately available

set -e

MAX_RETRIES=30
RETRY_INTERVAL=1
ATTEMPT=0

echo "🔄 Waiting for PostgreSQL to be available..."

while [ $ATTEMPT -lt $MAX_RETRIES ]; do
    if python -c "
import os
import django
from django.db import connections
from django.db.utils import OperationalError

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flipped_classroom_project.settings')
django.setup()

try:
    conn = connections['default']
    conn.ensure_connection()
    print('✅ Database is ready!')
    exit(0)
except OperationalError as e:
    print(f'Database not ready (attempt {$ATTEMPT + 1}/$MAX_RETRIES): {e}')
    exit(1)
" 2>/dev/null; then
        break
    fi
    
    ATTEMPT=$((ATTEMPT + 1))
    if [ $ATTEMPT -lt $MAX_RETRIES ]; then
        echo "⏳ Retrying in ${RETRY_INTERVAL}s... (${ATTEMPT}/${MAX_RETRIES})"
        sleep $RETRY_INTERVAL
    fi
done

if [ $ATTEMPT -ge $MAX_RETRIES ]; then
    echo "❌ Database connection failed after $MAX_RETRIES attempts. Exiting."
    exit 1
fi

echo "🚀 Running migrations..."
python manage.py migrate --noinput

echo "👤 Creating admin user..."
python manage.py create_admin

echo "✨ All done! Starting Gunicorn..."
exec gunicorn flipped_classroom_project.wsgi:application --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT
