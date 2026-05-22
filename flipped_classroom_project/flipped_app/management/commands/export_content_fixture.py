"""
Export subjects, videos, materials, quizzes (metadata) for loading on Render.

  python manage.py export_content_fixture
  python manage.py loaddata fixtures/fliplearn_content.json   # on Render shell
"""

import json
from pathlib import Path

from django.core.management import call_command
from django.core.management.base import BaseCommand


MODELS = [
    'flipped_app.Subject',
    'flipped_app.VideoLecture',
    'flipped_app.StudyMaterial',
    'flipped_app.Quiz',
    'flipped_app.QuizQuestion',
    'flipped_app.Assignment',
]


class Command(BaseCommand):
    help = 'Export platform content (videos, notes, quizzes) to a JSON fixture.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--output',
            default='fixtures/fliplearn_content.json',
            help='Output path relative to project root (default: fixtures/fliplearn_content.json)',
        )

    def handle(self, *args, **options):
        out = Path(options['output'])
        if not out.is_absolute():
            out = Path(__file__).resolve().parents[3] / out
        out.parent.mkdir(parents=True, exist_ok=True)

        with out.open('w', encoding='utf-8') as fh:
            call_command(
                'dumpdata',
                *MODELS,
                indent=2,
                stdout=fh,
                natural_foreign=True,
                natural_primary=True,
            )

        data = json.loads(out.read_text(encoding='utf-8'))
        self.stdout.write(self.style.SUCCESS(
            f'Exported {len(data)} records to {out}'
        ))
        self.stdout.write(
            'On Render: upload this file, then run:\n'
            '  python manage.py loaddata fixtures/fliplearn_content.json'
        )
