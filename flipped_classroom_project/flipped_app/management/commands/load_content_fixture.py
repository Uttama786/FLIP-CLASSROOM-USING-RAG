"""
Load videos, notes, quizzes from fixtures/fliplearn_content.json into the database.

Creates required uploader accounts first (admin, teacher, prof_sharma), then runs loaddata.
Used automatically on Render startup when the DB has little or no content.

  python manage.py load_content_fixture
  python manage.py load_content_fixture --force   # delete content models and reload
"""

import os
from pathlib import Path

from django.contrib.auth.models import User
from django.core.management import call_command
from django.core.management.base import BaseCommand

from flipped_app.models import (
    Assignment,
    Quiz,
    QuizQuestion,
    StudyMaterial,
    Subject,
    VideoLecture,
)

FIXTURE_PATH = Path(__file__).resolve().parents[3] / 'fixtures' / 'fliplearn_content.json'
MIN_VIDEOS_LOADED = 50  # skip reload if production already has content

REQUIRED_UPLOADERS = [
    ('admin', 'admin@fliplearn.local'),
    ('teacher', 'teacher@fliplearn.local'),
    ('prof_sharma', 'prof@fliplearn.local'),
]


def _ensure_uploaders(stdout, style):
    """Natural keys in the fixture reference these usernames."""
    created = 0
    for username, email in REQUIRED_UPLOADERS:
        user, was_created = User.objects.get_or_create(
            username=username,
            defaults={
                'email': email,
                'is_staff': True,
                'is_active': True,
            },
        )
        if was_created:
            # Unusable password — login via DJANGO_SUPERUSER admin only on Render
            user.set_unusable_password()
            user.save()
            created += 1
            stdout.write(style.SUCCESS(f'  Created uploader account: {username}'))
    if created == 0:
        stdout.write('  Uploader accounts already present.')
    return created


def _clear_content():
    QuizQuestion.objects.all().delete()
    Quiz.objects.all().delete()
    Assignment.objects.all().delete()
    StudyMaterial.objects.all().delete()
    VideoLecture.objects.all().delete()
    Subject.objects.all().delete()


class Command(BaseCommand):
    help = 'Load fliplearn_content.json (131 videos, 60 materials, quizzes) for production.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Delete existing subjects/videos/materials/quizzes then reload fixture.',
        )

    def handle(self, *args, **options):
        if not FIXTURE_PATH.is_file():
            self.stderr.write(self.style.ERROR(f'Fixture not found: {FIXTURE_PATH}'))
            return

        video_count = VideoLecture.objects.count()
        material_count = StudyMaterial.objects.count()

        if video_count >= MIN_VIDEOS_LOADED and not options['force']:
            self.stdout.write(self.style.SUCCESS(
                f'Content already loaded (videos={video_count}, materials={material_count}) — skipping.'
            ))
            return

        if video_count > 0 and not options['force']:
            self.stdout.write(self.style.WARNING(
                f'Partial content detected (videos={video_count}, materials={material_count}). '
                f'Run with --force to replace, or set FLIPLEARN_FORCE_RELOAD_CONTENT=true on Render.'
            ))
            return

        if options['force'] and video_count > 0:
            self.stdout.write('Clearing existing content …')
            _clear_content()

        self.stdout.write('Ensuring uploader accounts for fixture …')
        _ensure_uploaders(self.stdout, self.style)

        self.stdout.write(f'Loading {FIXTURE_PATH.name} …')
        try:
            call_command('loaddata', str(FIXTURE_PATH), verbosity=1)
        except Exception as exc:
            self.stderr.write(self.style.ERROR(f'loaddata failed: {exc}'))
            raise

        self.stdout.write(self.style.SUCCESS(
            f'Loaded. videos={VideoLecture.objects.count()} '
            f'materials={StudyMaterial.objects.count()} '
            f'subjects={Subject.objects.count()}'
        ))
