"""
Management command: migrate_with_retry
Runs Django migrations with automatic retry for Render cold starts.

Usage:
    python manage.py migrate_with_retry

This command is useful on platforms like Render where the database may take
a few seconds to become available during deployment.
"""

import time
from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.db import connections
from django.db.utils import OperationalError


class Command(BaseCommand):
    help = 'Run migrations with automatic retry for database unavailability.'

    def add_arguments(self, parser):
        parser.add_argument('--max-retries', type=int, default=30, 
                          help='Maximum number of retry attempts (default: 30)')
        parser.add_argument('--retry-interval', type=int, default=1,
                          help='Seconds to wait between retries (default: 1)')

    def wait_for_db(self, max_retries=30, retry_interval=1):
        """Wait for database to be available."""
        for attempt in range(max_retries):
            try:
                conn = connections['default']
                conn.ensure_connection()
                self.stdout.write(self.style.SUCCESS('[migrate_with_retry] Database is ready.'))
                return True
            except OperationalError as e:
                if attempt < max_retries - 1:
                    self.stdout.write(
                        self.style.WARNING(
                            f'[migrate_with_retry] Database not ready (attempt {attempt + 1}/{max_retries}). '
                            f'Retrying in {retry_interval}s...\nError: {e}'
                        )
                    )
                    time.sleep(retry_interval)
                else:
                    self.stdout.write(
                        self.style.ERROR(
                            f'[migrate_with_retry] Database connection failed after {max_retries} attempts:\n{e}'
                        )
                    )
                    return False
        return False

    def handle(self, *args, **options):
        max_retries = options.get('max_retries', 30)
        retry_interval = options.get('retry_interval', 1)

        # Wait for database before proceeding
        if not self.wait_for_db(max_retries=max_retries, retry_interval=retry_interval):
            self.stdout.write(self.style.ERROR('[migrate_with_retry] Aborting: database unavailable.'))
            raise SystemExit(1)

        try:
            self.stdout.write('[migrate_with_retry] Running migrations...')
            call_command('migrate', '--noinput', verbosity=1)
            self.stdout.write(self.style.SUCCESS('[migrate_with_retry] Migrations completed successfully.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'[migrate_with_retry] Migration failed: {e}'))
            raise SystemExit(1)
