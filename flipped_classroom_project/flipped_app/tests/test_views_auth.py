"""
Tests for Auth and Dashboard views.
Covers: home, register, login, logout, dashboard (student/teacher/admin roles).
"""

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from flipped_app.models import StudentProfile, Subject, TeacherProfile


def make_student(username="stu", password="pass12345"):
    user = User.objects.create_user(username=username, password=password)
    StudentProfile.objects.create(user=user, roll_number=f"R_{username}")
    return user


def make_teacher(username="tch", password="pass12345"):
    user = User.objects.create_user(username=username, password=password)
    TeacherProfile.objects.create(user=user, employee_id=f"E_{username}")
    return user


def make_admin(username="adm", password="pass12345"):
    return User.objects.create_superuser(
        username=username, email=f"{username}@ex.com", password=password
    )


# ──────────────────────────────────────────────────────────────
# Home View
# ──────────────────────────────────────────────────────────────

class HomeViewTest(TestCase):
    def test_unauthenticated_sees_home(self):
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "home.html")

    def test_authenticated_redirects_to_dashboard(self):
        user = make_student("home_stu")
        self.client.force_login(user)
        response = self.client.get(reverse("home"))
        self.assertRedirects(response, reverse("dashboard"))

    def test_home_contains_stats_keys(self):
        response = self.client.get(reverse("home"))
        self.assertIn("stats", response.context)
        self.assertIn("total_students", response.context["stats"])

    def test_home_contains_features_list(self):
        response = self.client.get(reverse("home"))
        self.assertIn("features_list", response.context)
        self.assertTrue(len(response.context["features_list"]) > 0)


# ──────────────────────────────────────────────────────────────
# Register View
# ──────────────────────────────────────────────────────────────

class RegisterViewTest(TestCase):
    def _reg_data(self, username="new_user", roll="NEW001"):
        return {
            "username": username,
            "first_name": "Alice",
            "last_name": "Wonder",
            "email": f"{username}@test.com",
            "password1": "SecurePass@123",
            "password2": "SecurePass@123",
            "roll_number": roll,
            "semester": 2,
            "phone": "9000000001",
            "previous_gpa": "7.0",
        }

    def test_get_register_page(self):
        response = self.client.get(reverse("register"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "register.html")

    def test_successful_registration_creates_user_and_profile(self):
        response = self.client.post(reverse("register"), data=self._reg_data())
        self.assertRedirects(response, reverse("dashboard"))
        self.assertTrue(User.objects.filter(username="new_user").exists())
        user = User.objects.get(username="new_user")
        self.assertTrue(hasattr(user, "student_profile"))

    def test_invalid_form_shows_errors(self):
        data = self._reg_data()
        data["password2"] = "wrongpass"
        response = self.client.post(reverse("register"), data=data)
        self.assertEqual(response.status_code, 200)  # stays on page
        self.assertFalse(User.objects.filter(username="new_user").exists())


# ──────────────────────────────────────────────────────────────
# Login View
# ──────────────────────────────────────────────────────────────

class LoginViewTest(TestCase):
    def setUp(self):
        self.user = make_student("login_stu")

    def test_get_login_page(self):
        response = self.client.get(reverse("login"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "login.html")

    def test_valid_login_redirects_to_dashboard(self):
        response = self.client.post(
            reverse("login"),
            data={"username": "login_stu", "password": "pass12345"},
        )
        self.assertRedirects(response, reverse("dashboard"))

    def test_invalid_login_stays_on_login_page(self):
        response = self.client.post(
            reverse("login"),
            data={"username": "login_stu", "password": "wrongpass"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "login.html")

    def test_authenticated_user_redirected_from_login(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("login"))
        self.assertRedirects(response, reverse("dashboard"))

    def test_login_clears_chat_history(self):
        from flipped_app.models import ChatMessage
        ChatMessage.objects.create(student=self.user, role="user", content="old message")
        self.assertEqual(ChatMessage.objects.filter(student=self.user).count(), 1)
        self.client.post(
            reverse("login"),
            data={"username": "login_stu", "password": "pass12345"},
        )
        self.assertEqual(ChatMessage.objects.filter(student=self.user).count(), 0)


# ──────────────────────────────────────────────────────────────
# Logout View
# ──────────────────────────────────────────────────────────────

class LogoutViewTest(TestCase):
    def setUp(self):
        self.user = make_student("logout_stu")
        self.client.force_login(self.user)

    def test_post_logout_redirects_to_login(self):
        response = self.client.post(reverse("logout"))
        self.assertRedirects(response, reverse("login"))

    def test_get_logout_redirects_to_dashboard(self):
        # GET to /logout/ should redirect to dashboard (not bounce loop)
        response = self.client.get(reverse("logout"))
        self.assertRedirects(response, reverse("dashboard"))

    def test_unauthenticated_logout_redirects(self):
        self.client.logout()
        response = self.client.get(reverse("logout"))
        self.assertEqual(response.status_code, 302)


# ──────────────────────────────────────────────────────────────
# Dashboard View — Student
# ──────────────────────────────────────────────────────────────

class StudentDashboardTest(TestCase):
    def setUp(self):
        self.student = make_student("dash_stu")
        self.client.force_login(self.student)

    def test_student_dashboard_status(self):
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "dashboard_student.html")

    def test_student_dashboard_role(self):
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.context["role"], "student")

    def test_student_dashboard_profile_in_context(self):
        response = self.client.get(reverse("dashboard"))
        self.assertIn("profile", response.context)


# ──────────────────────────────────────────────────────────────
# Dashboard View — Teacher
# ──────────────────────────────────────────────────────────────

class TeacherDashboardTest(TestCase):
    def setUp(self):
        self.teacher = make_teacher("dash_tch")
        self.client.force_login(self.teacher)

    def test_teacher_dashboard_status(self):
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "dashboard_teacher.html")

    def test_teacher_dashboard_role(self):
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.context["role"], "teacher")


# ──────────────────────────────────────────────────────────────
# Dashboard View — Admin
# ──────────────────────────────────────────────────────────────

class AdminDashboardTest(TestCase):
    def setUp(self):
        self.admin = make_admin("dash_adm")
        self.client.force_login(self.admin)

    def test_admin_dashboard_status(self):
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "dashboard_admin.html")

    def test_admin_dashboard_role(self):
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.context["role"], "admin")

    def test_admin_dashboard_stats_in_context(self):
        response = self.client.get(reverse("dashboard"))
        self.assertIn("total_students", response.context)
        self.assertIn("total_teachers", response.context)
        self.assertIn("at_risk_count", response.context)

    def test_admin_dashboard_perf_distribution_json(self):
        response = self.client.get(reverse("dashboard"))
        self.assertIn("perf_distribution", response.context)


# ──────────────────────────────────────────────────────────────
# Dashboard — No profile user redirected
# ──────────────────────────────────────────────────────────────

class DashboardNoProfileTest(TestCase):
    def test_user_without_profile_redirected(self):
        """A user with no student/teacher/admin profile is redirected away from dashboard."""
        bare_user = User.objects.create_user(username="bare", password="pass12345")
        self.client.force_login(bare_user)
        # Do NOT follow=True — a bare user causes a redirect loop (dashboard→login→dashboard)
        # We just verify the initial redirect happens (302)
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 302)
