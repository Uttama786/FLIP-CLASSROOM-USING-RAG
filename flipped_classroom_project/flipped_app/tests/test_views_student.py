"""
Tests for Student-facing views.
Covers: Subjects (list/enroll), Videos (list/detail/mark-watched),
Materials (list), Quizzes (list/take/result), Assignments (list/submit),
My-Performance, Notifications, Attendance (history/my), VideoComments.
"""

from datetime import date, timedelta

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from flipped_app.models import (
    Assignment,
    AssignmentSubmission,
    Attendance,
    Notification,
    Quiz,
    QuizAttempt,
    QuizQuestion,
    StudentPerformance,
    StudentProfile,
    Subject,
    VideoLecture,
    VideoWatchHistory,
)


def make_student(username="stu", password="pass12345"):
    user = User.objects.create_user(username=username, password=password)
    StudentProfile.objects.create(user=user, roll_number=f"R_{username}")
    return user


def make_subject(name="DS", code="DS"):
    obj, _ = Subject.objects.get_or_create(code=code, defaults={"name": name})
    return obj


# ──────────────────────────────────────────────────────────────
# Subject views
# ──────────────────────────────────────────────────────────────

class SubjectViewsTest(TestCase):
    def setUp(self):
        self.student = make_student("sub_stu")
        self.subject = make_subject()
        self.client.force_login(self.student)

    def test_subject_list_requires_login(self):
        self.client.logout()
        response = self.client.get(reverse("subjects"))
        self.assertEqual(response.status_code, 302)

    def test_subject_list_200(self):
        response = self.client.get(reverse("subjects"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "subjects.html")

    def test_enroll_adds_subject(self):
        self.client.post(reverse("enroll", args=[self.subject.id]))
        self.assertIn(self.subject, self.student.student_profile.enrolled_subjects.all())

    def test_enroll_requires_post(self):
        response = self.client.get(reverse("enroll", args=[self.subject.id]))
        self.assertRedirects(response, reverse("subjects"))

    def test_enroll_requires_login(self):
        self.client.logout()
        response = self.client.post(reverse("enroll", args=[self.subject.id]))
        self.assertEqual(response.status_code, 302)


# ──────────────────────────────────────────────────────────────
# Video views
# ──────────────────────────────────────────────────────────────

class VideoViewsTest(TestCase):
    def setUp(self):
        self.student = make_student("vid_stu")
        self.subject = make_subject("Video Subject", "VS")
        self.student.student_profile.enrolled_subjects.add(self.subject)
        self.video = VideoLecture.objects.create(
            subject=self.subject,
            title="Test Video",
            youtube_url="https://youtu.be/test123",
            duration_minutes=20,
        )
        self.client.force_login(self.student)

    def test_video_list_200(self):
        response = self.client.get(reverse("videos"))
        self.assertEqual(response.status_code, 200)

    def test_video_detail_200(self):
        response = self.client.get(reverse("video_detail", args=[self.video.id]))
        self.assertEqual(response.status_code, 200)

    def test_video_detail_creates_watch_history(self):
        self.client.get(reverse("video_detail", args=[self.video.id]))
        self.assertTrue(
            VideoWatchHistory.objects.filter(student=self.student, video=self.video).exists()
        )

    def test_mark_video_watched_post(self):
        response = self.client.post(reverse("mark_video_watched", args=[self.video.id]))
        self.assertEqual(response.status_code, 200)
        history = VideoWatchHistory.objects.get(student=self.student, video=self.video)
        self.assertTrue(history.completed)

    def test_mark_video_watched_get_returns_405(self):
        response = self.client.get(reverse("mark_video_watched", args=[self.video.id]))
        self.assertEqual(response.status_code, 405)

    def test_mark_video_watched_idempotent(self):
        # First mark
        self.client.post(reverse("mark_video_watched", args=[self.video.id]))
        # Second mark — should succeed but report already_completed=True
        import json
        response = self.client.post(reverse("mark_video_watched", args=[self.video.id]))
        data = json.loads(response.content)
        self.assertTrue(data["already_completed"])

    def test_video_list_by_subject(self):
        response = self.client.get(reverse("videos_by_subject", args=[self.subject.id]))
        self.assertEqual(response.status_code, 200)

    def test_requires_login(self):
        self.client.logout()
        response = self.client.get(reverse("videos"))
        self.assertEqual(response.status_code, 302)


# ──────────────────────────────────────────────────────────────
# Material views (student)
# ──────────────────────────────────────────────────────────────

class MaterialViewsStudentTest(TestCase):
    def setUp(self):
        self.student = make_student("mat_stu")
        self.subject = make_subject("Materials Subject", "MS")
        self.student.student_profile.enrolled_subjects.add(self.subject)
        self.client.force_login(self.student)

    def test_material_list_200(self):
        response = self.client.get(reverse("materials"))
        self.assertEqual(response.status_code, 200)

    def test_materials_by_subject(self):
        response = self.client.get(reverse("materials_by_subject", args=[self.subject.id]))
        self.assertEqual(response.status_code, 200)


# ──────────────────────────────────────────────────────────────
# Quiz views (student)
# ──────────────────────────────────────────────────────────────

class QuizViewsStudentTest(TestCase):
    def setUp(self):
        self.student = make_student("qz_stu")
        self.subject = make_subject("Quiz Subject", "QS")
        self.student.student_profile.enrolled_subjects.add(self.subject)
        self.quiz = Quiz.objects.create(
            subject=self.subject,
            title="Test Quiz",
            total_marks=10,
            is_active=True,
        )
        self.question = QuizQuestion.objects.create(
            quiz=self.quiz,
            question_text="What is BFS?",
            option_a="Breadth-First",
            option_b="Depth-First",
            option_c="Best-First",
            option_d="None",
            correct_answer="A",
            marks=10,
        )
        self.client.force_login(self.student)

    def test_quiz_list_200(self):
        response = self.client.get(reverse("quizzes"))
        self.assertEqual(response.status_code, 200)

    def test_take_quiz_get(self):
        response = self.client.get(reverse("take_quiz", args=[self.quiz.id]))
        self.assertEqual(response.status_code, 200)

    def test_take_quiz_correct_answer_scores(self):
        self.client.post(
            reverse("take_quiz", args=[self.quiz.id]),
            data={f"q_{self.question.id}": "A"},
        )
        attempt = QuizAttempt.objects.get(quiz=self.quiz, student=self.student)
        self.assertEqual(attempt.score, 10)

    def test_take_quiz_wrong_answer_zero_score(self):
        self.client.post(
            reverse("take_quiz", args=[self.quiz.id]),
            data={f"q_{self.question.id}": "B"},
        )
        attempt = QuizAttempt.objects.get(quiz=self.quiz, student=self.student)
        self.assertEqual(attempt.score, 0)

    def test_cannot_retake_quiz(self):
        QuizAttempt.objects.create(quiz=self.quiz, student=self.student, score=8)
        response = self.client.post(
            reverse("take_quiz", args=[self.quiz.id]),
            data={f"q_{self.question.id}": "A"},
        )
        # Should redirect to quiz_result (already attempted)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(QuizAttempt.objects.filter(quiz=self.quiz, student=self.student).count(), 1)

    def test_quiz_past_due_date_blocked(self):
        past_quiz = Quiz.objects.create(
            subject=self.subject,
            title="Past Quiz",
            total_marks=10,
            is_active=True,
            due_date=timezone.now() - timedelta(hours=1),
        )
        response = self.client.post(reverse("take_quiz", args=[past_quiz.id]), data={})
        self.assertRedirects(response, reverse("quizzes"))
        self.assertEqual(past_quiz.attempts.count(), 0)

    def test_quiz_result_view(self):
        QuizAttempt.objects.create(quiz=self.quiz, student=self.student, score=7)
        response = self.client.get(reverse("quiz_result", args=[self.quiz.id]))
        self.assertEqual(response.status_code, 200)
        self.assertIn("percentage", response.context)


# ──────────────────────────────────────────────────────────────
# Assignment views (student)
# ──────────────────────────────────────────────────────────────

class AssignmentViewsStudentTest(TestCase):
    def setUp(self):
        self.student = make_student("asgn_stu")
        self.subject = make_subject("Assignment Subject", "AS")
        self.student.student_profile.enrolled_subjects.add(self.subject)
        teacher = User.objects.create_user(username="asgn_tch", password="pass12345")
        self.assignment = Assignment.objects.create(
            subject=self.subject,
            title="Code Review",
            description="Review code",
            total_marks=20,
            due_date=timezone.now() + timedelta(days=7),
            created_by=teacher,
        )
        self.client.force_login(self.student)

    def test_assignment_list_200(self):
        response = self.client.get(reverse("assignments"))
        self.assertEqual(response.status_code, 200)

    def test_submit_assignment_get(self):
        response = self.client.get(reverse("submit_assignment", args=[self.assignment.id]))
        self.assertEqual(response.status_code, 200)

    def test_submit_assignment_post_creates_submission(self):
        dummy = SimpleUploadedFile("solution.pdf", b"data", content_type="application/pdf")
        response = self.client.post(
            reverse("submit_assignment", args=[self.assignment.id]),
            data={"submitted_file": dummy},
        )
        self.assertRedirects(response, reverse("assignments"))
        self.assertEqual(
            AssignmentSubmission.objects.filter(
                assignment=self.assignment, student=self.student
            ).count(),
            1,
        )

    def test_submit_past_due_date_blocked(self):
        past = Assignment.objects.create(
            subject=self.subject,
            title="Closed",
            description="",
            total_marks=10,
            due_date=timezone.now() - timedelta(hours=1),
        )
        response = self.client.post(reverse("submit_assignment", args=[past.id]), data={})
        self.assertRedirects(response, reverse("assignments"))
        self.assertEqual(past.submissions.count(), 0)

    def test_cannot_double_submit(self):
        dummy = SimpleUploadedFile("sol1.pdf", b"data", content_type="application/pdf")
        AssignmentSubmission.objects.create(
            assignment=self.assignment,
            student=self.student,
            submitted_file=dummy,
        )
        dummy2 = SimpleUploadedFile("sol2.pdf", b"data", content_type="application/pdf")
        self.client.post(
            reverse("submit_assignment", args=[self.assignment.id]),
            data={"submitted_file": dummy2},
        )
        self.assertEqual(
            AssignmentSubmission.objects.filter(
                assignment=self.assignment, student=self.student
            ).count(),
            1,
        )


# ──────────────────────────────────────────────────────────────
# My Performance view
# ──────────────────────────────────────────────────────────────

class MyPerformanceViewTest(TestCase):
    def setUp(self):
        self.student = make_student("perf_stu")
        self.client.force_login(self.student)

    def test_my_performance_200(self):
        response = self.client.get(reverse("my_performance"))
        self.assertEqual(response.status_code, 200)

    def test_requires_login(self):
        self.client.logout()
        response = self.client.get(reverse("my_performance"))
        self.assertEqual(response.status_code, 302)


# ──────────────────────────────────────────────────────────────
# Notification views
# ──────────────────────────────────────────────────────────────

class NotificationViewTest(TestCase):
    def setUp(self):
        self.student = make_student("notif_stu")
        self.notif = Notification.objects.create(
            recipient=self.student, message="Test notification"
        )
        self.client.force_login(self.student)

    def test_mark_notification_read(self):
        response = self.client.post(reverse("mark_read", args=[self.notif.id]))
        self.assertEqual(response.status_code, 302)
        self.notif.refresh_from_db()
        self.assertTrue(self.notif.is_read)


# ──────────────────────────────────────────────────────────────
# Attendance views (student)
# ──────────────────────────────────────────────────────────────

class AttendanceStudentViewTest(TestCase):
    def setUp(self):
        self.student = make_student("att_stu")
        self.subject = make_subject("Attendance Subj", "ATT")
        self.student.student_profile.enrolled_subjects.add(self.subject)
        Attendance.objects.create(
            student=self.student, subject=self.subject, date=date.today(), present=True
        )
        self.client.force_login(self.student)

    def test_my_attendance_200(self):
        response = self.client.get(reverse("my_attendance"))
        self.assertEqual(response.status_code, 200)

    def test_attendance_history_requires_login(self):
        self.client.logout()
        response = self.client.get(reverse("attendance_history"))
        self.assertEqual(response.status_code, 302)


# ──────────────────────────────────────────────────────────────
# Video Comment views
# ──────────────────────────────────────────────────────────────

class VideoCommentViewTest(TestCase):
    def setUp(self):
        self.student = make_student("cmt_stu")
        self.subject = make_subject("Comment Subject", "CS2")
        self.video = VideoLecture.objects.create(
            subject=self.subject,
            title="Comment Video",
            youtube_url="https://youtu.be/cmt",
        )
        self.client.force_login(self.student)

    def test_add_comment_post(self):
        response = self.client.post(
            reverse("add_video_comment", args=[self.video.id]),
            data={"body": "Great video!"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.video.comments.count(), 1)

    def test_add_empty_comment_ignored(self):
        response = self.client.post(
            reverse("add_video_comment", args=[self.video.id]),
            data={"body": ""},
        )
        # Should not create a comment with empty body
        self.assertEqual(self.video.comments.count(), 0)

    def test_delete_own_comment(self):
        self.client.post(
            reverse("add_video_comment", args=[self.video.id]),
            data={"body": "To delete"},
        )
        comment = self.video.comments.first()
        response = self.client.post(reverse("delete_video_comment", args=[comment.id]))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.video.comments.count(), 0)


# ──────────────────────────────────────────────────────────────
# Feedback view
# ──────────────────────────────────────────────────────────────

class FeedbackViewTest(TestCase):
    def setUp(self):
        self.student = make_student("fb_stu")
        self.client.force_login(self.student)

    def test_feedback_get_200(self):
        response = self.client.get(reverse("feedback"))
        self.assertEqual(response.status_code, 200)

    def test_feedback_post_creates_record(self):
        from flipped_app.models import Feedback
        response = self.client.post(
            reverse("feedback"),
            data={"category": "general", "rating": 5, "message": "Excellent platform!"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Feedback.objects.filter(author=self.student).count(), 1)

    def test_feedback_requires_login(self):
        self.client.logout()
        response = self.client.get(reverse("feedback"))
        self.assertEqual(response.status_code, 302)
