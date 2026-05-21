#!/usr/bin/env python
"""
Startup script: startup.py
Waits for PostgreSQL and runs migrations before starting Gunicorn.
More robust than shell scripts for cross-platform Render deployments.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Ensure Django settings module is discoverable
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flipped_classroom_project.settings')

MAX_RETRIES = 30
RETRY_INTERVAL = 1


def wait_for_database():
    """Wait for PostgreSQL to be available."""
    print("🔄 Waiting for PostgreSQL to be available...")
    
    for attempt in range(MAX_RETRIES):
        try:
            import django
            from django.conf import settings
            from django.db import connections
            
            # Configure Django
            django.setup()
            
            # Try to get a database connection
            conn = connections['default']
            with conn.cursor() as cursor:
                cursor.execute('SELECT 1')
            
            print("✅ Database is ready!")
            return True
            
        except Exception as e:
            attempt_num = attempt + 1
            print(f"⚠️  Database not ready (attempt {attempt_num}/{MAX_RETRIES}): {type(e).__name__}: {e}")
            
            if attempt_num < MAX_RETRIES:
                print(f"⏳ Retrying in {RETRY_INTERVAL}s...")
                time.sleep(RETRY_INTERVAL)
            else:
                print(f"❌ Database connection failed after {MAX_RETRIES} attempts.")
                return False
    
    return False


def run_command(cmd, description):
    """Run a shell command and exit if it fails."""
    print(f"\n🚀 {description}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"✅ {description} completed!")


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 FlipLearn Startup Script")
    print("=" * 60)
    
    # Wait for database
    if not wait_for_database():
        print("❌ Failed to connect to database. Exiting.")
        sys.exit(1)
    
    # Run migrations
    run_command('python manage.py migrate --noinput', 'Running migrations')
    
    # Create admin user
    run_command('python manage.py create_admin', 'Creating admin user')
    
    # Start Gunicorn
    print("\n✨ Starting Gunicorn...")
    port = os.environ.get('PORT', '8000')
    print(f"📍 Listening on 0.0.0.0:{port}")
    print("=" * 60)
    os.execvp('gunicorn', [
        'gunicorn',
        'flipped_classroom_project.wsgi:application',
        '--workers', '1',
        '--threads', '4',
        '--timeout', '120',
        '--bind', f'0.0.0.0:{port}'
    ])
