"""
Management command: create_admin
Creates a Django superuser from environment variables if one doesn't exist.

Usage (Procfile release step):
    python manage.py create_admin

Required env vars:
    DJANGO_SUPERUSER_USERNAME  (default: admin)
    DJANGO_SUPERUSER_EMAIL     (default: admin@example.com)
    DJANGO_SUPERUSER_PASSWORD  (required)
"""

import os
import time
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import connections
from django.db.utils import OperationalError


class Command(BaseCommand):
    help = 'Create a superuser from environment variables if none exists.'

    def wait_for_db(self, max_retries=30, retry_interval=1):
        """Wait for database to be available (useful for Render cold starts)."""
        for attempt in range(max_retries):
            try:
                conn = connections['default']
                conn.ensure_connection()
                self.stdout.write(self.style.SUCCESS('[create_admin] Database is ready.'))
                return True
            except OperationalError as e:
                if attempt < max_retries - 1:
                    self.stdout.write(
                        self.style.WARNING(
                            f'[create_admin] Database not ready (attempt {attempt + 1}/{max_retries}). '
                            f'Retrying in {retry_interval}s...'
                        )
                    )
                    time.sleep(retry_interval)
                else:
                    self.stdout.write(
                        self.style.ERROR(
                            f'[create_admin] Database connection failed after {max_retries} attempts: {e}'
                        )
                    )
                    return False
        return False

    def handle(self, *args, **options):
        # Wait for database before proceeding
        if not self.wait_for_db():
            self.stdout.write(self.style.ERROR('[create_admin] Aborting: database unavailable.'))
            return

        User = get_user_model()
        if User.objects.filter(is_superuser=True).exists():
            self.stdout.write('[create_admin] Superuser already exists — skipping.')
            return

        username = os.environ.get('DJANGO_SUPERUSER_USERNAME', 'admin').strip()
        email = os.environ.get('DJANGO_SUPERUSER_EMAIL', 'admin@example.com').strip()
        password = os.environ.get('DJANGO_SUPERUSER_PASSWORD', '').strip()

        if not password:
            self.stdout.write(
                self.style.WARNING(
                    '[create_admin] DJANGO_SUPERUSER_PASSWORD not set — skipping superuser creation.'
                )
            )
            return

        if len(password) < 12:
            self.stdout.write(
                self.style.WARNING(
                    '[create_admin] Password too short (<12 chars) — skipping superuser creation.'
                )
            )
            return

        User.objects.create_superuser(username=username, email=email, password=password)
        self.stdout.write(
            self.style.SUCCESS(f'[create_admin] Superuser "{username}" created successfully.')
        )
