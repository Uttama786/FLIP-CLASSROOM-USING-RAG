"""
Tests for flipped_app/forms.py
Covers all forms: SubjectForm, StudentRegistrationForm, VideoLectureForm,
StudyMaterialForm, QuizForm, QuizQuestionForm, AssignmentForm,
AssignmentSubmissionForm, GradeSubmissionForm.
"""

from datetime import timedelta

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.utils import timezone

from flipped_app.forms import (
    AssignmentForm,
    AssignmentSubmissionForm,
    GradeSubmissionForm,
    QuizForm,
    QuizQuestionForm,
    StudentRegistrationForm,
    StudyMaterialForm,
    SubjectForm,
    VideoLectureForm,
)
from flipped_app.models import (
    Assignment,
    AssignmentSubmission,
    Quiz,
    Subject,
)


# ──────────────────────────────────────────────────────────────
# SubjectForm
# ──────────────────────────────────────────────────────────────

class SubjectFormTest(TestCase):
    def test_valid_form(self):
        # Use unique test-only codes that won't conflict with seeded data
        form = SubjectForm(data={"name": "Operating Systems Test", "code": "OS99", "description": ""})
        self.assertTrue(form.is_valid())
        self.assertEqual(form.cleaned_data["code"], "OS99")  # uppercased

    def test_code_uppercased(self):
        # Use a unique code not in seed data
        form = SubjectForm(data={"name": "Python Test", "code": "py99"})
        self.assertTrue(form.is_valid())
        self.assertEqual(form.cleaned_data["code"], "PY99")

    def test_name_required(self):
        form = SubjectForm(data={"name": "", "code": "XX99"})
        self.assertFalse(form.is_valid())
        self.assertIn("name", form.errors)

    def test_code_required(self):
        form = SubjectForm(data={"name": "Test", "code": ""})
        self.assertFalse(form.is_valid())
        self.assertIn("code", form.errors)


# ──────────────────────────────────────────────────────────────
# StudentRegistrationForm
# ──────────────────────────────────────────────────────────────

class StudentRegistrationFormTest(TestCase):
    def _valid_data(self, username="stu1", roll="R001"):
        return {
            "username": username,
            "first_name": "Alice",
            "last_name": "Smith",
            "email": f"{username}@test.com",
            "password1": "SecurePass@123",
            "password2": "SecurePass@123",
            "roll_number": roll,
            "semester": 3,
            "phone": "9876543210",
            "previous_gpa": "8.5",
        }

    def test_valid_registration(self):
        form = StudentRegistrationForm(data=self._valid_data())
        self.assertTrue(form.is_valid(), form.errors)

    def test_saves_student_profile(self):
        form = StudentRegistrationForm(data=self._valid_data("stu2", "R002"))
        self.assertTrue(form.is_valid())
        user = form.save()
        self.assertTrue(hasattr(user, "student_profile"))
        self.assertEqual(user.student_profile.roll_number, "R002")

    def test_duplicate_email_rejected(self):
        User.objects.create_user(username="existing", email="stu1@test.com", password="pass")
        form = StudentRegistrationForm(data=self._valid_data())
        self.assertFalse(form.is_valid())
        self.assertIn("email", form.errors)

    def test_duplicate_roll_number_rejected(self):
        # Create the first user & profile
        form = StudentRegistrationForm(data=self._valid_data("stu_a", "R999"))
        self.assertTrue(form.is_valid())
        form.save()

        # Try the same roll number again
        form2 = StudentRegistrationForm(data=self._valid_data("stu_b", "R999"))
        self.assertFalse(form2.is_valid())
        self.assertIn("roll_number", form2.errors)

    def test_password_mismatch_rejected(self):
        data = self._valid_data()
        data["password2"] = "DifferentPass@456"
        form = StudentRegistrationForm(data=data)
        self.assertFalse(form.is_valid())

    def test_email_stored_lowercase(self):
        data = self._valid_data("stu_case", "R_CASE")
        data["email"] = "StuCase@Test.COM"
        form = StudentRegistrationForm(data=data)
        self.assertTrue(form.is_valid(), form.errors)
        user = form.save()
        self.assertEqual(user.email, "stucase@test.com")

    def test_semester_out_of_range(self):
        data = self._valid_data("stu3", "R003")
        data["semester"] = 9
        form = StudentRegistrationForm(data=data)
        self.assertFalse(form.is_valid())

    def test_previous_gpa_out_of_range(self):
        data = self._valid_data("stu4", "R004")
        data["previous_gpa"] = "11.0"
        form = StudentRegistrationForm(data=data)
        self.assertFalse(form.is_valid())


# ──────────────────────────────────────────────────────────────
# VideoLectureForm
# ──────────────────────────────────────────────────────────────

class VideoLectureFormTest(TestCase):
    def setUp(self):
        self.subject, _ = Subject.objects.get_or_create(code="DS", defaults={"name": "Data Structures"})

    def test_valid_with_youtube_url(self):
        form = VideoLectureForm(data={
            "subject": self.subject.id,
            "title": "BFS Lecture",
            "description": "Intro",
            "youtube_url": "https://youtu.be/abc",
            "duration_minutes": 30,
        })
        self.assertTrue(form.is_valid(), form.errors)

    def test_invalid_without_video_or_url(self):
        form = VideoLectureForm(data={
            "subject": self.subject.id,
            "title": "Missing Source",
            "description": "",
            "youtube_url": "",
            "duration_minutes": 10,
        })
        self.assertFalse(form.is_valid())

    def test_invalid_video_extension_rejected(self):
        bad_file = SimpleUploadedFile("video.exe", b"data", content_type="application/octet-stream")
        form = VideoLectureForm(
            data={
                "subject": self.subject.id,
                "title": "Bad Video",
                "description": "",
                "youtube_url": "",
                "duration_minutes": 5,
            },
            files={"video_file": bad_file},
        )
        self.assertFalse(form.is_valid())
        self.assertIn("video_file", form.errors)

    def test_valid_mp4_file(self):
        mp4 = SimpleUploadedFile("lecture.mp4", b"fake mp4 data", content_type="video/mp4")
        form = VideoLectureForm(
            data={
                "subject": self.subject.id,
                "title": "MP4 Lecture",
                "description": "",
                "youtube_url": "",
                "duration_minutes": 45,
            },
            files={"video_file": mp4},
        )
        self.assertTrue(form.is_valid(), form.errors)


# ──────────────────────────────────────────────────────────────
# StudyMaterialForm
# ──────────────────────────────────────────────────────────────

class StudyMaterialFormTest(TestCase):
    def setUp(self):
        self.subject, _ = Subject.objects.get_or_create(code="PY", defaults={"name": "Python"})

    def _upload_form(self, filename, content_type="application/pdf"):
        f = SimpleUploadedFile(filename, b"dummy content", content_type=content_type)
        return StudyMaterialForm(
            data={"subject": self.subject.id, "title": "Notes", "description": ""},
            files={"file": f},
        )

    def test_valid_pdf_upload(self):
        form = self._upload_form("notes.pdf")
        self.assertTrue(form.is_valid(), form.errors)

    def test_valid_docx_upload(self):
        form = self._upload_form(
            "notes.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        self.assertTrue(form.is_valid(), form.errors)

    def test_valid_txt_upload(self):
        form = self._upload_form("notes.txt", content_type="text/plain")
        self.assertTrue(form.is_valid(), form.errors)

    def test_exe_file_rejected(self):
        form = self._upload_form("payload.exe", content_type="application/octet-stream")
        self.assertFalse(form.is_valid())
        self.assertIn("Unsupported file type", str(form.errors))

    def test_missing_title_rejected(self):
        f = SimpleUploadedFile("a.pdf", b"data", content_type="application/pdf")
        form = StudyMaterialForm(
            data={"subject": self.subject.id, "title": "", "description": ""},
            files={"file": f},
        )
        self.assertFalse(form.is_valid())

    def test_missing_subject_rejected(self):
        f = SimpleUploadedFile("a.pdf", b"data", content_type="application/pdf")
        form = StudyMaterialForm(
            data={"subject": "", "title": "Notes", "description": ""},
            files={"file": f},
        )
        self.assertFalse(form.is_valid())


# ──────────────────────────────────────────────────────────────
# QuizForm
# ──────────────────────────────────────────────────────────────

class QuizFormTest(TestCase):
    def setUp(self):
        self.subject, _ = Subject.objects.get_or_create(code="CN", defaults={"name": "Computer Networks"})

    def test_valid_quiz_form(self):
        form = QuizForm(data={
            "subject": self.subject.id,
            "title": "Midterm Quiz",
            "description": "Test",
            "total_marks": 20,
            "time_limit_minutes": 30,
            "due_date": "",
        })
        self.assertTrue(form.is_valid(), form.errors)

    def test_missing_title_rejected(self):
        form = QuizForm(data={
            "subject": self.subject.id,
            "title": "",
            "total_marks": 10,
            "time_limit_minutes": 20,
        })
        self.assertFalse(form.is_valid())


# ──────────────────────────────────────────────────────────────
# QuizQuestionForm
# ──────────────────────────────────────────────────────────────

class QuizQuestionFormTest(TestCase):
    def _valid_data(self):
        return {
            "question_text": "What is a stack?",
            "option_a": "LIFO",
            "option_b": "FIFO",
            "option_c": "Random",
            "option_d": "None",
            "correct_answer": "A",
            "marks": 2,
        }

    def test_valid_question(self):
        form = QuizQuestionForm(data=self._valid_data())
        self.assertTrue(form.is_valid(), form.errors)

    def test_invalid_correct_answer_choice(self):
        data = self._valid_data()
        data["correct_answer"] = "E"  # Not a valid choice
        form = QuizQuestionForm(data=data)
        self.assertFalse(form.is_valid())

    def test_missing_question_text(self):
        data = self._valid_data()
        data["question_text"] = ""
        form = QuizQuestionForm(data=data)
        self.assertFalse(form.is_valid())


# ──────────────────────────────────────────────────────────────
# AssignmentForm
# ──────────────────────────────────────────────────────────────

class AssignmentFormTest(TestCase):
    def setUp(self):
        self.subject, _ = Subject.objects.get_or_create(code="WD", defaults={"name": "Web Development"})

    def _valid_data(self):
        future = (timezone.now() + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M")
        return {
            "subject": self.subject.id,
            "title": "Build a REST API",
            "description": "Use Django",
            "total_marks": 30,
            "due_date": future,
        }

    def test_valid_assignment(self):
        form = AssignmentForm(data=self._valid_data())
        self.assertTrue(form.is_valid(), form.errors)

    def test_attachment_exe_rejected(self):
        data = self._valid_data()
        bad_file = SimpleUploadedFile("hack.exe", b"data")
        form = AssignmentForm(data=data, files={"attachment": bad_file})
        self.assertFalse(form.is_valid())

    def test_pdf_attachment_accepted(self):
        data = self._valid_data()
        pdf = SimpleUploadedFile("brief.pdf", b"data", content_type="application/pdf")
        form = AssignmentForm(data=data, files={"attachment": pdf})
        self.assertTrue(form.is_valid(), form.errors)


# ──────────────────────────────────────────────────────────────
# AssignmentSubmissionForm
# ──────────────────────────────────────────────────────────────

class AssignmentSubmissionFormTest(TestCase):
    def test_valid_pdf_submission(self):
        f = SimpleUploadedFile("solution.pdf", b"data", content_type="application/pdf")
        form = AssignmentSubmissionForm(files={"submitted_file": f})
        self.assertTrue(form.is_valid(), form.errors)

    def test_valid_py_submission(self):
        f = SimpleUploadedFile("solution.py", b"print('hello')", content_type="text/x-python")
        form = AssignmentSubmissionForm(files={"submitted_file": f})
        self.assertTrue(form.is_valid(), form.errors)

    def test_exe_submission_rejected(self):
        f = SimpleUploadedFile("virus.exe", b"data")
        form = AssignmentSubmissionForm(files={"submitted_file": f})
        self.assertFalse(form.is_valid())

    def test_java_submission_accepted(self):
        f = SimpleUploadedFile("Main.java", b"class Main{}", content_type="text/x-java-source")
        form = AssignmentSubmissionForm(files={"submitted_file": f})
        self.assertTrue(form.is_valid(), form.errors)


# ──────────────────────────────────────────────────────────────
# GradeSubmissionForm
# ──────────────────────────────────────────────────────────────

class GradeSubmissionFormTest(TestCase):
    def setUp(self):
        subject, _ = Subject.objects.get_or_create(code="AIML", defaults={"name": "AI & ML"})
        teacher = User.objects.create_user(username="teacher_gf", password="pass")
        assignment = Assignment.objects.create(
            subject=subject,
            title="Assignment",
            description="",
            total_marks=50,
            due_date=timezone.now() + timedelta(days=1),
            created_by=teacher,
        )
        student = User.objects.create_user(username="student_gf", password="pass")
        from django.core.files.uploadedfile import SimpleUploadedFile
        self.submission = AssignmentSubmission.objects.create(
            assignment=assignment,
            student=student,
            submitted_file=SimpleUploadedFile("sol.pdf", b"data"),
        )

    def test_valid_grade(self):
        form = GradeSubmissionForm(
            data={"marks_obtained": 40, "feedback": "Good work!"},
            instance=self.submission,
            total_marks=50,
        )
        self.assertTrue(form.is_valid(), form.errors)

    def test_negative_marks_rejected(self):
        form = GradeSubmissionForm(
            data={"marks_obtained": -1, "feedback": ""},
            instance=self.submission,
            total_marks=50,
        )
        self.assertFalse(form.is_valid())

    def test_marks_exceed_total_rejected(self):
        form = GradeSubmissionForm(
            data={"marks_obtained": 60, "feedback": ""},
            instance=self.submission,
            total_marks=50,
        )
        self.assertFalse(form.is_valid())
        self.assertIn("marks_obtained", form.errors)

    def test_exact_total_marks_valid(self):
        form = GradeSubmissionForm(
            data={"marks_obtained": 50, "feedback": "Perfect!"},
            instance=self.submission,
            total_marks=50,
        )
        self.assertTrue(form.is_valid(), form.errors)
