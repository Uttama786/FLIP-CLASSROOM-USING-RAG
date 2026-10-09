import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flipped_classroom_project.settings')
django.setup()

from django.test import TestCase
from django.contrib.auth.models import User
from flipped_app.models import StudentProfile, TeacherProfile
from ml_model.academic_intelligence import (
    get_engine,
    handle_academic_query,
    extract_student_target,
    detect_focus_area,
)


class AcademicIntelligenceTests(TestCase):
    def setUp(self):
        self.engine = get_engine()
        # Create users for RBAC testing
        self.teacher_user = User.objects.create_user(
            username='test_prof', email='prof@example.com', password='password123', is_staff=True
        )
        TeacherProfile.objects.create(user=self.teacher_user, employee_id='EMP999')

        self.student_nisha = User.objects.create_user(
            username='test_nisha', email='nisha@example.com', password='password123'
        )
        StudentProfile.objects.create(
            user=self.student_nisha, roll_number='2024IT0001', department='IT', semester=4
        )

        self.student_other = User.objects.create_user(
            username='test_other', email='other@example.com', password='password123'
        )
        StudentProfile.objects.create(
            user=self.student_other, roll_number='2025CS9999', department='CSE', semester=2
        )

    def test_multi_record_retrieval(self):
        """2024IT0001 has 2 records (PY and CN) in dataset.csv."""
        records = self.engine.find_student_records('2024IT0001')
        self.assertGreaterEqual(len(records), 2)
        usns = {r['usn'] for r in records}
        self.assertEqual(usns, {'2024IT0001'})

    def test_student_profile_aggregation(self):
        """Aggregation computes means and totals properly."""
        records = self.engine.find_student_records('2024IT0001')
        profile = self.engine.aggregate_student_profile(records)
        self.assertEqual(profile['usn'], '2024IT0001')
        self.assertEqual(profile['student_name'], 'Nisha Mishra')
        self.assertEqual(profile['record_count'], len(records))
        self.assertGreater(profile['avg_quiz_score'], 0)
        self.assertGreater(profile['avg_attendance'], 0)
        self.assertGreater(profile['total_video_time_minutes'], 0)

    def test_ml_prediction_no_data_leakage(self):
        """ML prediction works and input features do NOT include final_exam_score."""
        records = self.engine.find_student_records('2024IT0001')
        profile = self.engine.aggregate_student_profile(records)
        pred = self.engine.predict_performance(profile)
        self.assertIn('predicted_score', pred)
        self.assertIn('predicted_label', pred)
        self.assertNotIn('final_exam_score', pred['features_used'])
        self.assertTrue(0 <= pred['predicted_score'] <= 100)

    def test_risk_evaluation_and_factors(self):
        """Risk assessment is reproducible and identifies risk factors."""
        analysis = self.engine.analyze_student('2025EE0005')
        self.assertTrue(analysis['found'])
        risk = analysis['risk']
        self.assertIn(risk['badge'], ['HIGH RISK', 'AT RISK', 'MODERATE RISK', 'LOW RISK'])
        if risk['badge'] in ('HIGH RISK', 'AT RISK'):
            self.assertGreater(len(risk['main_risk_factors']), 0)
            self.assertGreater(len(risk['recommended_interventions']), 0)

    def test_career_recommendations(self):
        """Career recommendations output non-empty domains with suitability."""
        analysis = self.engine.analyze_student('2024IT0001')
        careers = analysis['careers']
        self.assertGreaterEqual(len(careers), 1)
        for c in careers:
            self.assertIn('domain', c)
            self.assertIn(c['suitability'], ['Strong suitability', 'Moderate suitability'])

    def test_cohort_queries(self):
        """Cohort queries return matching student lists."""
        high_risk = self.engine.query_cohort('high_risk')
        self.assertIsInstance(high_risk, list)
        att_low = self.engine.query_cohort('attendance_below', 75.0)
        self.assertIsInstance(att_low, list)
        for s in att_low:
            self.assertLess(s['attendance'], 75.0)

    def test_rbac_student_denied_other_student(self):
        """Student cannot query another student's marks or records."""
        res = handle_academic_query('Show me Rahul\'s marks', self.student_other)
        self.assertTrue(res['handled'])
        self.assertIn('Access Denied', res['reply'])

        res2 = handle_academic_query('Show details of USN 2024IT0001', self.student_other)
        self.assertTrue(res2['handled'])
        self.assertIn('Access Denied', res2['reply'])

    def test_rbac_student_denied_cohort_list(self):
        """Student cannot view cohort-wide risk lists."""
        res = handle_academic_query('Show all high-risk students', self.student_other)
        self.assertTrue(res['handled'])
        self.assertIn('Access Denied', res['reply'])

    def test_rbac_student_allowed_own_data(self):
        """Student querying themselves is permitted."""
        res = handle_academic_query('How am I performing?', self.student_nisha)
        self.assertTrue(res['handled'])
        self.assertNotIn('Access Denied', res['reply'])
        self.assertIn('Nisha Mishra', res['reply'])

    def test_rbac_teacher_allowed_full_access(self):
        """Teacher can query any student and cohort lists."""
        res = handle_academic_query('Show me the details of USN 2024IT0001', self.teacher_user)
        self.assertTrue(res['handled'])
        self.assertNotIn('Access Denied', res['reply'])
        self.assertIn('Nisha Mishra', res['reply'])

        res_cohort = handle_academic_query('Show all high-risk students', self.teacher_user)
        self.assertTrue(res_cohort['handled'])
        self.assertNotIn('Access Denied', res_cohort['reply'])
        self.assertIn('HIGH-RISK STUDENTS', res_cohort['reply'])

    def test_nonexistent_student(self):
        """Unknown USN returns unavailable message."""
        res = handle_academic_query('Show details of USN 9999ZZ9999', self.teacher_user)
        self.assertTrue(res['handled'])
        self.assertIn('The required information is not available in the current dataset', res['reply'])
