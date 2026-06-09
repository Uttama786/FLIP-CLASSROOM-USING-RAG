"""
Tests for Teacher/Admin-facing views.
Covers: Upload Video, Upload Material, Create Quiz, Add Question,
Create Assignment, Grade Submission, Assignment Submissions,
Analytics, Student Accounts, Add Subject, Delete Subject,
Attendance (mark/history), Run ML, Export CSV.
"""

from datetime import timedelta

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from flipped_app.models import (
    Assignment,
    AssignmentSubmission,
    Attendance,
    Quiz,
    QuizQuestion,
    StudentPerformance,
    StudentProfile,
    Subject,
    TeacherProfile,
    VideoLecture,
)


def make_teacher(username="tch", password="pass12345"):
    user = User.objects.create_user(username=username, password=password)
    TeacherProfile.objects.create(user=user, employee_id=f"E_{username}")
    return user


def make_student(username="stu", password="pass12345"):
    user = User.objects.create_user(username=username, password=password)
    StudentProfile.objects.create(user=user, roll_number=f"R_{username}")
    return user


def make_admin(username="adm", password="pass12345"):
    return User.objects.create_superuser(
        username=username, email=f"{username}@ex.com", password=password
    )


def make_subject(name="DS", code="DS"):
    obj, _ = Subject.objects.get_or_create(code=code, defaults={"name": name})
    return obj


# ──────────────────────────────────────────────────────────────
# Upload Video
# ──────────────────────────────────────────────────────────────

class UploadVideoViewTest(TestCase):
    def setUp(self):
        self.teacher = make_teacher("vid_tch")
        self.subject = make_subject("Video Sub", "VS2")
        self.client.force_login(self.teacher)

    def test_get_upload_page(self):
        response = self.client.get(reverse("upload_video"))
        self.assertEqual(response.status_code, 200)

    def test_post_creates_video_with_youtube_url(self):
        response = self.client.post(reverse("upload_video"), data={
            "subject": self.subject.id,
            "title": "New Video",
            "description": "Desc",
            "youtube_url": "https://youtu.be/newvid",
            "duration_minutes": 45,
        })
        self.assertRedirects(response, reverse("videos"))
        self.assertTrue(VideoLecture.objects.filter(title="New Video").exists())

    def test_student_cannot_upload_video(self):
        student = make_student("vid_stu_block")
        self.client.force_login(student)
        response = self.client.get(reverse("upload_video"))
        self.assertEqual(response.status_code, 302)

    def test_unauthenticated_redirected(self):
        self.client.logout()
        response = self.client.get(reverse("upload_video"))
        self.assertEqual(response.status_code, 302)


# ──────────────────────────────────────────────────────────────
# Upload Material
# ──────────────────────────────────────────────────────────────

class UploadMaterialViewTest(TestCase):
    def setUp(self):
        self.teacher = make_teacher("mat_tch")
        self.subject = make_subject("Material Sub", "MTS")
        self.client.force_login(self.teacher)

    def test_get_upload_material_page(self):
        response = self.client.get(reverse("upload_material"))
        self.assertEqual(response.status_code, 200)

    def test_post_creates_material(self):
        pdf = SimpleUploadedFile("notes.pdf", b"pdf content", content_type="application/pdf")
        response = self.client.post(reverse("upload_material"), data={
            "subject": self.subject.id,
            "title": "Lecture Notes",
            "description": "Notes",
            "file": pdf,
        })
        self.assertRedirects(response, reverse("materials"))

    def test_student_cannot_upload_material(self):
        student = make_student("mat_stu_block")
        self.client.force_login(student)
        response = self.client.get(reverse("upload_material"))
        self.assertEqual(response.status_code, 302)


# ──────────────────────────────────────────────────────────────
# Create Quiz
# ──────────────────────────────────────────────────────────────

class CreateQuizViewTest(TestCase):
    def setUp(self):
        self.teacher = make_teacher("quiz_tch")
        self.subject = make_subject("Quiz Sub", "QZ2")
        self.client.force_login(self.teacher)

    def test_get_create_quiz_page(self):
        response = self.client.get(reverse("create_quiz"))
        self.assertEqual(response.status_code, 200)

    def test_post_creates_quiz(self):
        response = self.client.post(reverse("create_quiz"), data={
            "subject": self.subject.id,
            "title": "New Quiz",
            "description": "Test quiz",
            "total_marks": 20,
            "time_limit_minutes": 30,
            "due_date": "",
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Quiz.objects.filter(title="New Quiz").exists())

    def test_student_cannot_create_quiz(self):
        student = make_student("quiz_stu_block")
        self.client.force_login(student)
        response = self.client.get(reverse("create_quiz"))
        self.assertEqual(response.status_code, 302)


# ──────────────────────────────────────────────────────────────
# Add Question
# ──────────────────────────────────────────────────────────────

class AddQuestionViewTest(TestCase):
    def setUp(self):
        self.teacher = make_teacher("q_tch")
        self.subject = make_subject("Q Sub", "QSB")
        self.quiz = Quiz.objects.create(subject=self.subject, title="Sample Quiz", total_marks=10)
        self.client.force_login(self.teacher)

    def test_get_add_question_page(self):
        response = self.client.get(reverse("add_question", args=[self.quiz.id]))
        self.assertEqual(response.status_code, 200)

    def test_post_adds_question(self):
        self.client.post(reverse("add_question", args=[self.quiz.id]), data={
            "question_text": "What is DFS?",
            "option_a": "Depth-First",
            "option_b": "Breadth-First",
            "option_c": "Best-First",
            "option_d": "None",
            "correct_answer": "A",
            "marks": 2,
        })
        self.assertEqual(self.quiz.questions.count(), 1)


# ──────────────────────────────────────────────────────────────
# Create Assignment
# ──────────────────────────────────────────────────────────────

class CreateAssignmentViewTest(TestCase):
    def setUp(self):
        self.teacher = make_teacher("asgn_tch")
        self.subject = make_subject("Assign Sub", "AS2")
        self.client.force_login(self.teacher)

    def test_get_create_assignment_page(self):
        response = self.client.get(reverse("create_assignment"))
        self.assertEqual(response.status_code, 200)

    def test_post_creates_assignment(self):
        future = (timezone.now() + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M")
        response = self.client.post(reverse("create_assignment"), data={
            "subject": self.subject.id,
            "title": "Homework 1",
            "description": "Do the work",
            "total_marks": 25,
            "due_date": future,
        })
        self.assertRedirects(response, reverse("assignments"))
        self.assertTrue(Assignment.objects.filter(title="Homework 1").exists())

    def test_student_cannot_create_assignment(self):
        student = make_student("asgn_stu_block")
        self.client.force_login(student)
        response = self.client.get(reverse("create_assignment"))
        self.assertEqual(response.status_code, 302)


# ──────────────────────────────────────────────────────────────
# Grade Submission
# ──────────────────────────────────────────────────────────────

class GradeSubmissionViewTest(TestCase):
    def setUp(self):
        self.teacher = make_teacher("grade_tch")
        self.subject = make_subject("Grade Sub", "GS")
        self.assignment = Assignment.objects.create(
            subject=self.subject,
            title="Grade Me",
            description="",
            total_marks=50,
            due_date=timezone.now() + timedelta(days=1),
            created_by=self.teacher,
        )
        self.student = make_student("grade_stu")
        self.submission = AssignmentSubmission.objects.create(
            assignment=self.assignment,
            student=self.student,
            submitted_file=SimpleUploadedFile("sol.pdf", b"data"),
        )
        self.client.force_login(self.teacher)

    def test_get_grade_page(self):
        response = self.client.get(reverse("grade_submission", args=[self.submission.id]))
        self.assertEqual(response.status_code, 200)

    def test_post_grades_submission(self):
        response = self.client.post(
            reverse("grade_submission", args=[self.submission.id]),
            data={"marks_obtained": 40, "feedback": "Good work!"},
        )
        self.assertEqual(response.status_code, 302)
        self.submission.refresh_from_db()
        self.assertTrue(self.submission.is_graded)
        self.assertEqual(self.submission.marks_obtained, 40)

    def test_invalid_marks_not_graded(self):
        self.client.post(
            reverse("grade_submission", args=[self.submission.id]),
            data={"marks_obtained": 999, "feedback": "Too many"},
        )
        self.submission.refresh_from_db()
        self.assertFalse(self.submission.is_graded)


# ──────────────────────────────────────────────────────────────
# Assignment Submissions
# ──────────────────────────────────────────────────────────────

class AssignmentSubmissionsViewTest(TestCase):
    def setUp(self):
        self.teacher = make_teacher("subs_tch")
        self.subject = make_subject("Sub Sub", "SS")
        self.assignment = Assignment.objects.create(
            subject=self.subject,
            title="View Subs",
            description="",
            total_marks=20,
            due_date=timezone.now() + timedelta(days=1),
            created_by=self.teacher,
        )
        self.client.force_login(self.teacher)

    def test_submissions_list_200(self):
        response = self.client.get(
            reverse("assignment_submissions", args=[self.assignment.id])
        )
        self.assertEqual(response.status_code, 200)

    def test_student_blocked_from_submissions(self):
        student = make_student("subs_stu_block")
        self.client.force_login(student)
        response = self.client.get(
            reverse("assignment_submissions", args=[self.assignment.id])
        )
        self.assertEqual(response.status_code, 302)


# ──────────────────────────────────────────────────────────────
# Analytics views
# ──────────────────────────────────────────────────────────────

class AnalyticsViewTest(TestCase):
    def setUp(self):
        self.teacher = make_teacher("analytics_tch")
        self.client.force_login(self.teacher)

    def test_analytics_200(self):
        response = self.client.get(reverse("analytics"))
        self.assertEqual(response.status_code, 200)

    def test_student_blocked_from_analytics(self):
        student = make_student("analytics_stu_block")
        self.client.force_login(student)
        response = self.client.get(reverse("analytics"))
        self.assertEqual(response.status_code, 302)

    def test_analytics_filter_by_subject(self):
        subject = make_subject("Filter Sub", "FS")
        response = self.client.get(f"{reverse('analytics')}?subject={subject.id}")
        self.assertEqual(response.status_code, 200)

    def test_analytics_filter_risk_only(self):
        response = self.client.get(f"{reverse('analytics')}?risk_only=1")
        self.assertEqual(response.status_code, 200)


# ──────────────────────────────────────────────────────────────
# Add Subject view
# ──────────────────────────────────────────────────────────────

class AddSubjectViewTest(TestCase):
    def setUp(self):
        self.teacher = make_teacher("add_sub_tch")
        self.client.force_login(self.teacher)

    def test_get_add_subject_page(self):
        response = self.client.get(reverse("add_subject"))
        self.assertEqual(response.status_code, 200)

    def test_post_creates_subject(self):
        response = self.client.post(reverse("add_subject"), data={
            "name": "Machine Learning",
            "code": "ML",
            "description": "",
        })
        self.assertRedirects(response, reverse("subjects"))
        self.assertTrue(Subject.objects.filter(code="ML").exists())

    def test_student_cannot_add_subject(self):
        student = make_student("add_sub_stu_block")
        self.client.force_login(student)
        response = self.client.get(reverse("add_subject"))
        self.assertEqual(response.status_code, 302)


# ──────────────────────────────────────────────────────────────
# Delete Subject view
# ──────────────────────────────────────────────────────────────

class DeleteSubjectViewTest(TestCase):
    def setUp(self):
        self.teacher = make_teacher("del_sub_tch")
        self.subject = make_subject("To Delete", "TD")
        self.client.force_login(self.teacher)

    def test_get_confirmation_page(self):
        response = self.client.get(reverse("delete_subject", args=[self.subject.id]))
        self.assertEqual(response.status_code, 200)

    def test_post_deletes_subject(self):
        response = self.client.post(reverse("delete_subject", args=[self.subject.id]))
        self.assertRedirects(response, reverse("subjects"))
        self.assertFalse(Subject.objects.filter(pk=self.subject.pk).exists())

    def test_student_cannot_delete_subject(self):
        student = make_student("del_sub_stu_block")
        self.client.force_login(student)
        response = self.client.post(reverse("delete_subject", args=[self.subject.id]))
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Subject.objects.filter(pk=self.subject.pk).exists())


# ──────────────────────────────────────────────────────────────
# Mark Attendance
# ──────────────────────────────────────────────────────────────

class MarkAttendanceViewTest(TestCase):
    def setUp(self):
        self.teacher = make_teacher("att_tch")
        self.subject = make_subject("Attendance Sub", "ATB")
        self.student = make_student("att_mk_stu")
        self.client.force_login(self.teacher)

    def test_mark_attendance_200(self):
        response = self.client.get(reverse("mark_attendance"))
        self.assertEqual(response.status_code, 200)

    def test_post_marks_attendance(self):
        from datetime import date
        response = self.client.post(reverse("mark_attendance"), data={
            "subject": self.subject.id,
            "date": date.today().isoformat(),
            f"attendance_{self.student.id}": "present",
        })
        self.assertEqual(response.status_code, 302)


# ──────────────────────────────────────────────────────────────
# Student Accounts view
# ──────────────────────────────────────────────────────────────

class StudentAccountsViewTest(TestCase):
    def setUp(self):
        self.teacher = make_teacher("accts_tch")
        make_student("accts_stu")
        self.client.force_login(self.teacher)

    def test_student_accounts_200(self):
        response = self.client.get(reverse("student_accounts"))
        self.assertEqual(response.status_code, 200)

    def test_student_blocked(self):
        student = make_student("accts_stu_block")
        self.client.force_login(student)
        response = self.client.get(reverse("student_accounts"))
        self.assertEqual(response.status_code, 302)


# ──────────────────────────────────────────────────────────────
# Export CSV
# ──────────────────────────────────────────────────────────────

class ExportCSVViewTest(TestCase):
    def setUp(self):
        self.teacher = make_teacher("csv_tch")
        self.client.force_login(self.teacher)

    def test_export_csv_returns_csv(self):
        response = self.client.get(reverse("export_csv"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.get("Content-Type", ""))

    def test_student_cannot_export(self):
        student = make_student("csv_stu_block")
        self.client.force_login(student)
        response = self.client.get(reverse("export_csv"))
        self.assertEqual(response.status_code, 302)


# ──────────────────────────────────────────────────────────────
# Student Detail Analytics
# ──────────────────────────────────────────────────────────────

class StudentDetailAnalyticsTest(TestCase):
    def setUp(self):
        self.teacher = make_teacher("detail_tch")
        self.student = make_student("detail_stu")
        self.client.force_login(self.teacher)

    def test_student_detail_analytics_200(self):
        response = self.client.get(
            reverse("student_analytics", args=[self.student.id])
        )
        self.assertEqual(response.status_code, 200)

    def test_student_blocked(self):
        self.client.force_login(self.student)
        response = self.client.get(
            reverse("student_analytics", args=[self.student.id])
        )
        self.assertEqual(response.status_code, 302)
