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
import psycopg2
from urllib.parse import urlparse

# Ensure Django settings module is discoverable
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flipped_classroom_project.settings')

MAX_RETRIES = 30
RETRY_INTERVAL = 2  # Increased from 1 to 2 seconds


def parse_database_url(db_url):
    """Parse DATABASE_URL into psycopg2 connection parameters."""
    if not db_url:
        return None
    
    # Handle postgresql:// and postgres:// schemes
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    
    parsed = urlparse(db_url)
    
    return {
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username,
        'password': parsed.password,
    }


def wait_for_database_raw():
    """
    Wait for PostgreSQL using raw psycopg2 connection.
    This is more reliable than Django's connection pool.
    """
    print("🔄 Waiting for PostgreSQL to be available...")
    
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("⚠️  DATABASE_URL not set, will use SQLite")
        return True
    
    db_params = parse_database_url(db_url)
    if not db_params:
        print("⚠️  Could not parse DATABASE_URL, will attempt connection")
        return True
    
    print(f"📍 Connecting to {db_params['host']}:{db_params['port']}/{db_params['database']}")
    
    for attempt in range(MAX_RETRIES):
        try:
            conn = psycopg2.connect(
                host=db_params['host'],
                port=db_params['port'],
                database=db_params['database'],
                user=db_params['user'],
                password=db_params['password'],
                connect_timeout=5
            )
            conn.close()
            print("✅ Database connection successful!")
            return True
            
        except psycopg2.OperationalError as e:
            attempt_num = attempt + 1
            error_msg = str(e).split('\n')[0]  # Get first line of error
            print(f"⚠️  Attempt {attempt_num}/{MAX_RETRIES}: {error_msg}")
            
            if attempt_num < MAX_RETRIES:
                print(f"⏳ Retrying in {RETRY_INTERVAL}s...")
                time.sleep(RETRY_INTERVAL)
        except Exception as e:
            attempt_num = attempt + 1
            print(f"⚠️  Attempt {attempt_num}/{MAX_RETRIES}: {type(e).__name__}: {e}")
            
            if attempt_num < MAX_RETRIES:
                print(f"⏳ Retrying in {RETRY_INTERVAL}s...")
                time.sleep(RETRY_INTERVAL)
    
    print(f"❌ Failed to connect to database after {MAX_RETRIES} attempts.")
    return False


def run_command(cmd, description):
    """Run a shell command and exit if it fails."""
    print(f"\n🚀 {description}...")
    print(f"   Command: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"✅ {description} completed!")


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 FlipLearn Startup Script")
    print("=" * 70)
    
    # Wait for database (raw connection test)
    if not wait_for_database_raw():
        print("❌ Database unavailable. Exiting.")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✅ Database is ready. Starting Django initialization...")
    print("=" * 70)
    
    # Run migrations
    run_command('python manage.py migrate --noinput', 'Running migrations')
    
    # Create admin user
    run_command('python manage.py create_admin', 'Creating admin user')
    
    # Start Gunicorn
    print("\n" + "=" * 70)
    print("✨ Starting Gunicorn...")
    port = os.environ.get('PORT', '8000')
    print(f"📍 Listening on 0.0.0.0:{port}")
    print("=" * 70 + "\n")
    os.execvp('gunicorn', [
        'gunicorn',
        'flipped_classroom_project.wsgi:application',
        '--workers', '1',
        '--threads', '4',
        '--timeout', '120',
        '--bind', f'0.0.0.0:{port}'
    ])
