from datetime import timedelta
from unittest.mock import patch

from django.conf import settings
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from flipped_app.forms import StudyMaterialForm
from flipped_app.models import Assignment, Quiz, StudentProfile, Subject


class AccessAndDueDateTests(TestCase):
    def setUp(self):
        self.subject, _ = Subject.objects.get_or_create(
            code="DS",
            defaults={"name": "Data Structures"},
        )

        self.student = User.objects.create_user(username="student1", password="pass12345")
        StudentProfile.objects.create(
            user=self.student,
            roll_number="R001",
            semester=1,
            previous_gpa=7.5,
        )

        self.teacher = User.objects.create_user(username="teacher1", password="pass12345")
        self.admin = User.objects.create_superuser(
            username="admin1",
            email="admin@example.com",
            password="pass12345",
        )

    def test_take_quiz_blocks_past_due_date(self):
        quiz = Quiz.objects.create(
            subject=self.subject,
            title="Closed Quiz",
            total_marks=10,
            due_date=timezone.now() - timedelta(hours=1),
            is_active=True,
        )

        self.client.force_login(self.student)
        response = self.client.post(reverse("take_quiz", args=[quiz.id]), data={})

        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse("quizzes"))
        self.assertEqual(quiz.attempts.count(), 0)

    def test_submit_assignment_blocks_past_due_date(self):
        assignment = Assignment.objects.create(
            subject=self.subject,
            title="Closed Assignment",
            description="desc",
            total_marks=20,
            due_date=timezone.now() - timedelta(hours=1),
        )

        self.client.force_login(self.student)
        response = self.client.post(reverse("submit_assignment", args=[assignment.id]), data={})

        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse("assignments"))
        self.assertEqual(assignment.submissions.count(), 0)

    @override_settings(RAG_REBUILD_SYNC=True)
    @patch("rag_engine.indexer.build_index")
    def test_rebuild_rag_requires_superuser(self, mock_build_index):
        self.client.force_login(self.teacher)
        denied = self.client.post(reverse("rebuild_rag"))
        self.assertEqual(denied.status_code, 403)
        mock_build_index.assert_not_called()

        self.client.force_login(self.admin)
        allowed = self.client.post(reverse("rebuild_rag"))
        self.assertEqual(allowed.status_code, 200)
        mock_build_index.assert_called_once()


class StaticSettingsSanityTests(TestCase):
    def test_staticfiles_dirs_is_empty_to_avoid_duplicate_collection(self):
        self.assertEqual(settings.STATICFILES_DIRS, [])

    def test_required_finders_present(self):
        self.assertIn("django.contrib.staticfiles.finders.FileSystemFinder", settings.STATICFILES_FINDERS)
        self.assertIn("django.contrib.staticfiles.finders.AppDirectoriesFinder", settings.STATICFILES_FINDERS)


class UploadAndErrorSanitizationTests(TestCase):
    def setUp(self):
        self.subject, _ = Subject.objects.get_or_create(
            code="DS",
            defaults={"name": "Data Structures"},
        )

    def test_study_material_form_rejects_unsafe_file_extension(self):
        unsafe = SimpleUploadedFile(
            "payload.exe",
            b"dummy",
            content_type="application/octet-stream",
        )
        form = StudyMaterialForm(
            data={
                "subject": self.subject.id,
                "title": "Bad Upload",
                "description": "Should fail",
            },
            files={"file": unsafe},
        )

        self.assertFalse(form.is_valid())
        self.assertIn("Unsupported file type", str(form.errors))

    @patch("rag_engine.chat._get_groq_client", side_effect=RuntimeError("sensitive backend detail"))
    @patch("rag_engine.chat.get_context", return_value=[
        {"text": "BFS is Breadth-First Search algorithm.", "source": "DS notes", "subject": "DS", "score": 0.9}
    ])
    def test_rag_ask_does_not_leak_internal_errors(self, _mock_ctx, _mock_client):
        from rag_engine.chat import ask

        result = ask("What is BFS?")
        self.assertEqual(result.get("error"), "internal_error")
        self.assertNotIn("sensitive backend detail", result.get("reply", ""))
