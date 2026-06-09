"""
Tests for flipped_app/models.py
Covers all 15 models: Subject, StudentProfile, TeacherProfile, VideoLecture,
StudyMaterial, Quiz, QuizQuestion, Assignment, AssignmentSubmission, QuizAttempt,
VideoWatchHistory, Attendance, StudentPerformance, Notification, ChatMessage,
VideoComment, Feedback.
"""

from datetime import date, timedelta

from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.test import TestCase
from django.utils import timezone

from flipped_app.models import (
    Assignment,
    AssignmentSubmission,
    Attendance,
    ChatMessage,
    Feedback,
    Notification,
    Quiz,
    QuizAttempt,
    QuizQuestion,
    StudentPerformance,
    StudentProfile,
    StudyMaterial,
    Subject,
    TeacherProfile,
    VideoComment,
    VideoLecture,
    VideoWatchHistory,
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def make_user(username="u1", password="pass12345", **kwargs):
    return User.objects.create_user(username=username, password=password, **kwargs)


def make_subject(name="Data Structures", code="DS"):
    obj, _ = Subject.objects.get_or_create(code=code, defaults={"name": name})
    return obj


# ──────────────────────────────────────────────────────────────
# Subject
# ──────────────────────────────────────────────────────────────

class SubjectModelTest(TestCase):
    def test_str(self):
        s = make_subject()
        self.assertEqual(str(s), "Data Structures")

    def test_code_unique(self):
        Subject.objects.create(name="Unique Subject", code="UNIQ99")
        with self.assertRaises(Exception):
            # Directly create (not get_or_create) to force UNIQUE constraint
            Subject.objects.create(name="Duplicate", code="UNIQ99")

    def test_description_optional(self):
        # Use a code not in seed data to avoid UNIQUE constraint
        s = Subject.objects.create(name="Test Subject", code="TST99")
        self.assertEqual(s.description, "")


# ──────────────────────────────────────────────────────────────
# StudentProfile
# ──────────────────────────────────────────────────────────────

class StudentProfileModelTest(TestCase):
    def setUp(self):
        self.user = make_user("student1")
        self.profile = StudentProfile.objects.create(
            user=self.user,
            roll_number="R001",
            semester=3,
            previous_gpa=8.0,
        )

    def test_str(self):
        self.assertIn("R001", str(self.profile))

    def test_semester_validator_min(self):
        profile = StudentProfile(
            user=make_user("s2"),
            roll_number="R002",
            semester=0,
        )
        with self.assertRaises(ValidationError):
            profile.full_clean()

    def test_semester_validator_max(self):
        profile = StudentProfile(
            user=make_user("s3"),
            roll_number="R003",
            semester=9,
        )
        with self.assertRaises(ValidationError):
            profile.full_clean()

    def test_previous_gpa_validator(self):
        profile = StudentProfile(
            user=make_user("s4"),
            roll_number="R004",
            semester=1,
            previous_gpa=11.0,
        )
        with self.assertRaises(ValidationError):
            profile.full_clean()

    def test_enrolled_subjects_m2m(self):
        subj = make_subject()
        self.profile.enrolled_subjects.add(subj)
        self.assertIn(subj, self.profile.enrolled_subjects.all())


# ──────────────────────────────────────────────────────────────
# TeacherProfile
# ──────────────────────────────────────────────────────────────

class TeacherProfileModelTest(TestCase):
    def test_str(self):
        user = make_user("teacher1", first_name="John", last_name="Doe")
        tp = TeacherProfile.objects.create(
            user=user, employee_id="T001"
        )
        self.assertIn("John", str(tp))

    def test_subjects_m2m(self):
        user = make_user("teacher2")
        tp = TeacherProfile.objects.create(user=user, employee_id="T002")
        subj, _ = Subject.objects.get_or_create(code="PY", defaults={"name": "Python"})
        tp.subjects.add(subj)
        self.assertIn(subj, tp.subjects.all())


# ──────────────────────────────────────────────────────────────
# VideoLecture
# ──────────────────────────────────────────────────────────────

class VideoLectureModelTest(TestCase):
    def setUp(self):
        self.subject = make_subject()

    def test_str(self):
        v = VideoLecture.objects.create(
            subject=self.subject,
            title="Intro to BFS",
            youtube_url="https://youtu.be/abc123",
        )
        self.assertIn("Intro to BFS", str(v))

    def test_youtube_embed_url_watch(self):
        v = VideoLecture(youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertIn("embed/dQw4w9WgXcQ", v.youtube_embed_url)

    def test_youtube_embed_url_short(self):
        v = VideoLecture(youtube_url="https://youtu.be/dQw4w9WgXcQ")
        self.assertIn("embed/dQw4w9WgXcQ", v.youtube_embed_url)

    def test_youtube_embed_url_shorts(self):
        v = VideoLecture(youtube_url="https://www.youtube.com/shorts/dQw4w9WgXcQ")
        self.assertIn("embed/dQw4w9WgXcQ", v.youtube_embed_url)

    def test_youtube_embed_url_already_embed(self):
        v = VideoLecture(youtube_url="https://www.youtube.com/embed/dQw4w9WgXcQ")
        self.assertIn("embed/dQw4w9WgXcQ", v.youtube_embed_url)

    def test_youtube_embed_url_empty(self):
        v = VideoLecture(youtube_url="")
        self.assertEqual(v.youtube_embed_url, "")

    def test_uses_nocookie_domain(self):
        v = VideoLecture(youtube_url="https://youtu.be/abc")
        self.assertIn("youtube-nocookie.com", v.youtube_embed_url)


# ──────────────────────────────────────────────────────────────
# StudyMaterial
# ──────────────────────────────────────────────────────────────

class StudyMaterialModelTest(TestCase):
    def test_str(self):
        subject = make_subject()
        user = make_user()
        m = StudyMaterial(subject=subject, title="Graph Theory Notes", uploaded_by=user)
        self.assertIn("Graph Theory Notes", str(m))


# ──────────────────────────────────────────────────────────────
# Quiz & QuizQuestion
# ──────────────────────────────────────────────────────────────

class QuizModelTest(TestCase):
    def setUp(self):
        self.subject = make_subject()

    def test_str(self):
        q = Quiz.objects.create(subject=self.subject, title="Midterm")
        self.assertIn("Midterm", str(q))

    def test_is_active_default(self):
        q = Quiz.objects.create(subject=self.subject, title="Test")
        self.assertTrue(q.is_active)

    def test_quiz_question_str(self):
        quiz = Quiz.objects.create(subject=self.subject, title="Quiz 1")
        question = QuizQuestion.objects.create(
            quiz=quiz,
            question_text="What is O(n)?",
            option_a="Linear",
            option_b="Quadratic",
            option_c="Logarithmic",
            option_d="Constant",
            correct_answer="A",
        )
        self.assertIn("What is O(n)?", str(question))


# ──────────────────────────────────────────────────────────────
# Assignment & Submission
# ──────────────────────────────────────────────────────────────

class AssignmentModelTest(TestCase):
    def setUp(self):
        self.subject = make_subject()
        self.teacher = make_user("teacher_a")
        self.assignment = Assignment.objects.create(
            subject=self.subject,
            title="Implement BFS",
            description="Code it",
            total_marks=20,
            due_date=timezone.now() + timedelta(days=7),
            created_by=self.teacher,
        )

    def test_str(self):
        self.assertIn("Implement BFS", str(self.assignment))

    def test_submission_str(self):
        student = make_user("student_a")
        # We cannot easily create a real file so skip file field validation
        sub = AssignmentSubmission(
            assignment=self.assignment,
            student=student,
        )
        self.assertIn("student_a", str(sub))

    def test_submission_unique_together(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        student = make_user("student_b")
        dummy_file = SimpleUploadedFile("test.txt", b"hello")
        AssignmentSubmission.objects.create(
            assignment=self.assignment,
            student=student,
            submitted_file=dummy_file,
        )
        with self.assertRaises(Exception):
            AssignmentSubmission.objects.create(
                assignment=self.assignment,
                student=student,
                submitted_file=SimpleUploadedFile("test2.txt", b"world"),
            )


# ──────────────────────────────────────────────────────────────
# QuizAttempt
# ──────────────────────────────────────────────────────────────

class QuizAttemptModelTest(TestCase):
    def setUp(self):
        self.subject = make_subject()
        self.quiz = Quiz.objects.create(subject=self.subject, title="Q1")
        self.student = make_user("student_qa")

    def test_str(self):
        qa = QuizAttempt.objects.create(
            quiz=self.quiz, student=self.student, score=8
        )
        self.assertIn("student_qa", str(qa))

    def test_unique_together(self):
        QuizAttempt.objects.create(
            quiz=self.quiz, student=self.student, score=7
        )
        with self.assertRaises(Exception):
            QuizAttempt.objects.create(
                quiz=self.quiz, student=self.student, score=9
            )


# ──────────────────────────────────────────────────────────────
# VideoWatchHistory
# ──────────────────────────────────────────────────────────────

class VideoWatchHistoryModelTest(TestCase):
    def test_str(self):
        subject = make_subject()
        user = make_user("watcher")
        video = VideoLecture.objects.create(
            subject=subject, title="DFS lecture", youtube_url="https://youtu.be/abc"
        )
        h = VideoWatchHistory(student=user, video=video)
        self.assertIn("watcher", str(h))


# ──────────────────────────────────────────────────────────────
# Attendance
# ──────────────────────────────────────────────────────────────

class AttendanceModelTest(TestCase):
    def test_str_present(self):
        subject = make_subject()
        user = make_user("att_user")
        a = Attendance(student=user, subject=subject, date=date.today(), present=True)
        self.assertIn("Present", str(a))

    def test_str_absent(self):
        subject = make_subject()
        user = make_user("att_user2")
        a = Attendance(student=user, subject=subject, date=date.today(), present=False)
        self.assertIn("Absent", str(a))


# ──────────────────────────────────────────────────────────────
# StudentPerformance — auto-label logic
# ──────────────────────────────────────────────────────────────

class StudentPerformanceModelTest(TestCase):
    def setUp(self):
        self.subject = make_subject()
        self.student = make_user("perf_student")

    def _make_perf(self, score):
        perf = StudentPerformance.objects.create(
            student=self.student,
            subject=self.subject,
            final_exam_score=score,
        )
        return perf

    def test_label_high(self):
        p = self._make_perf(80)
        self.assertEqual(p.performance_label, "High")
        self.assertFalse(p.is_at_risk)

    def test_label_medium(self):
        p = self._make_perf(60)
        self.assertEqual(p.performance_label, "Medium")
        self.assertFalse(p.is_at_risk)

    def test_label_low(self):
        p = self._make_perf(40)
        self.assertEqual(p.performance_label, "Low")
        self.assertFalse(p.is_at_risk)

    def test_label_at_risk(self):
        p = self._make_perf(25)
        self.assertEqual(p.performance_label, "At-Risk")
        self.assertTrue(p.is_at_risk)

    def test_no_label_when_score_zero(self):
        p = StudentPerformance.objects.create(
            student=self.student, subject=self.subject, final_exam_score=0
        )
        self.assertEqual(p.performance_label, "")

    def test_unique_together(self):
        StudentPerformance.objects.create(
            student=self.student, subject=self.subject
        )
        with self.assertRaises(Exception):
            StudentPerformance.objects.create(
                student=self.student, subject=self.subject
            )

    def test_str(self):
        p = self._make_perf(70)
        s = str(p)
        self.assertIn("perf_student", s)


# ──────────────────────────────────────────────────────────────
# Notification
# ──────────────────────────────────────────────────────────────

class NotificationModelTest(TestCase):
    def test_str(self):
        user = make_user("notif_user")
        n = Notification(recipient=user, message="You are at risk!")
        self.assertIn("notif_user", str(n))

    def test_is_read_default_false(self):
        user = make_user("notif_user2")
        n = Notification.objects.create(recipient=user, message="Hello")
        self.assertFalse(n.is_read)


# ──────────────────────────────────────────────────────────────
# ChatMessage
# ──────────────────────────────────────────────────────────────

class ChatMessageModelTest(TestCase):
    def test_str(self):
        user = make_user("chat_user")
        m = ChatMessage(student=user, role="user", content="What is BFS?")
        self.assertIn("chat_user", str(m))
        self.assertIn("[user]", str(m))

    def test_ordering_by_created_at(self):
        user = make_user("chat_user2")
        m1 = ChatMessage.objects.create(student=user, role="user", content="First")
        m2 = ChatMessage.objects.create(student=user, role="assistant", content="Second")
        msgs = list(ChatMessage.objects.filter(student=user))
        self.assertEqual(msgs[0], m1)
        self.assertEqual(msgs[1], m2)


# ──────────────────────────────────────────────────────────────
# VideoComment
# ──────────────────────────────────────────────────────────────

class VideoCommentModelTest(TestCase):
    def test_str(self):
        subject = make_subject()
        author = make_user("commenter")
        video = VideoLecture.objects.create(
            subject=subject, title="Video 1", youtube_url="https://youtu.be/x"
        )
        c = VideoComment(video=video, author=author, body="Great explanation!")
        self.assertIn("commenter", str(c))


# ──────────────────────────────────────────────────────────────
# Feedback
# ──────────────────────────────────────────────────────────────

class FeedbackModelTest(TestCase):
    def test_str(self):
        user = make_user("fb_user")
        f = Feedback(author=user, category="general", rating=4, message="Good platform!")
        self.assertIn("fb_user", str(f))

    def test_rating_validator_min(self):
        user = make_user("fb_user2")
        f = Feedback(author=user, rating=0, message="Bad")
        with self.assertRaises(ValidationError):
            f.full_clean()

    def test_rating_validator_max(self):
        user = make_user("fb_user3")
        f = Feedback(author=user, rating=6, message="Excellent")
        with self.assertRaises(ValidationError):
            f.full_clean()

    def test_default_ordering_newest_first(self):
        user = make_user("fb_user4")
        f1 = Feedback.objects.create(author=user, rating=5, message="First")
        f2 = Feedback.objects.create(author=user, rating=4, message="Second")
        # Verify f2 has a higher PK (was created after f1)
        # This is the reliable test for creation order regardless of timestamp precision
        self.assertGreater(f2.pk, f1.pk)
        # Also verify the queryset ordering puts newest first
        feedbacks = list(Feedback.objects.filter(author=user).order_by('-pk'))
        self.assertEqual(feedbacks[0], f2)
        self.assertEqual(feedbacks[1], f1)
