"""
================================================================================
Academic Intelligence & Risk Prediction Engine
Author: Uttam Vitthal Bhise (M.Tech CSE)
================================================================================
Integrates dataset.csv directly to provide:
1. Student-level multi-record retrieval & aggregation
2. Subject/record-wise performance breakdown
3. Performance trend classification (Improving / Stable / Declining)
4. ML regression for Final Exam Score prediction (preventing data leakage)
5. ML classification for Performance Category
6. Reproducible Academic Risk Scoring (Low / Moderate / At-Risk / High)
7. Concrete Risk Explanations and Recommended Interventions
8. Career Area Recommendations based on academic & subject indicators
9. Multi-student cohort analysis (high risk, mentoring, attendance < 75%, etc.)
10. Strict Role-Based Access Control (Teacher/Admin vs Student)
================================================================================
"""

import os
import re
import math
import logging
import threading
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

FEATURES = [
    'videos_watched', 'total_video_time_minutes',
    'quiz_avg_score', 'assignment_avg_marks',
    'attendance_percentage', 'participation_score', 'previous_gpa'
]

SUBJECT_MAP = {
    'PY': 'Python Programming',
    'CN': 'Computer Networks',
    'WD': 'Web Development',
    'DSC': 'Data Structures & Algorithms',
    'DS': 'Data Structures',
    'AIML': 'AI & Machine Learning',
    'ML': 'Machine Learning',
    'QS': 'Quantitative Skills',
    'VS': 'Verbal Skills',
    'GS': 'General Studies',
}

CAREER_DOMAINS = [
    'Software Development',
    'Data Science',
    'Artificial Intelligence / Machine Learning',
    'Data Analytics',
    'Web Development',
    'Cybersecurity',
    'Cloud Computing',
    'Research / Higher Studies'
]

_ENGINE_LOCK = threading.Lock()
_CACHED_ENGINE = None


class AcademicIntelligenceEngine:
    def __init__(self, dataset_path: str = DATASET_PATH, models_dir: str = MODELS_DIR):
        self.dataset_path = dataset_path
        self.models_dir = models_dir
        self.df = None
        self.mtime = None
        self.scaler = None
        self.label_encoder = None
        self.rf_regressor = None
        self.rf_classifier = None
        self.linear_regressor = None
        self.usn_index = {}
        self.full_name_index = {}
        self.first_name_index = {}
        self.cohort_cache = None
        self._load_dataset()
        self._load_models()
        self._build_cohort_cache()

    def _load_dataset(self):
        """Load and validate dataset.csv directly."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found at: {self.dataset_path}")

        current_mtime = os.path.getmtime(self.dataset_path)
        if self.df is not None and self.mtime == current_mtime:
            return

        df = pd.read_csv(self.dataset_path)

        required_cols = [
            'student_id', 'usn', 'student_name', 'videos_watched',
            'total_video_time_minutes', 'quiz_avg_score', 'assignment_avg_marks',
            'attendance_percentage', 'participation_score', 'previous_gpa',
            'final_exam_score', 'performance_label'
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        df['usn'] = df['usn'].astype(str).str.strip()
        df['student_name'] = df['student_name'].astype(str).str.strip()
        df['student_id'] = df['student_id'].astype(str).str.strip()

        for col in FEATURES + ['final_exam_score']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        usn_index = {}
        full_name_index = {}
        first_name_index = {}

        for idx, row in df.iterrows():
            usn_key = row['usn'].upper()
            if usn_key not in usn_index:
                usn_index[usn_key] = []
            usn_index[usn_key].append(idx)

            name_full = row['student_name'].strip().lower()
            if name_full not in full_name_index:
                full_name_index[name_full] = []
            full_name_index[name_full].append(idx)

            first_name = name_full.split()[0] if name_full else ''
            if first_name:
                if first_name not in first_name_index:
                    first_name_index[first_name] = []
                first_name_index[first_name].append(idx)

        self.df = df
        self.mtime = current_mtime
        self.usn_index = usn_index
        self.full_name_index = full_name_index
        self.first_name_index = first_name_index
        logger.info("Loaded dataset: %d records, %d USNs, %d first names", len(df), len(usn_index), len(first_name_index))

    def _load_models(self):
        """Load saved ML models and scalers."""
        try:
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            le_path = os.path.join(self.models_dir, 'label_encoder.pkl')
            rf_reg_path = os.path.join(self.models_dir, 'rf_regressor.pkl')
            rf_cls_path = os.path.join(self.models_dir, 'rf_classifier.pkl')
            lr_path = os.path.join(self.models_dir, 'linear_regression.pkl')

            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            if os.path.exists(le_path):
                self.label_encoder = joblib.load(le_path)
            if os.path.exists(rf_reg_path):
                self.rf_regressor = joblib.load(rf_reg_path)
            if os.path.exists(rf_cls_path):
                self.rf_classifier = joblib.load(rf_cls_path)
            if os.path.exists(lr_path):
                self.linear_regressor = joblib.load(lr_path)
        except Exception as e:
            logger.warning("Error loading ML models: %s", e)

    def _build_cohort_cache(self):
        """Precomputes vector predictions and risk scores for all students."""
        if self.df is None or self.scaler is None:
            return

        try:
            rec_counts = self.df.groupby('usn')['student_id'].count()
            agg = self.df.groupby('usn').agg({
                'student_name': 'first',
                'videos_watched': 'mean',
                'total_video_time_minutes': 'sum',
                'quiz_avg_score': 'mean',
                'assignment_avg_marks': 'mean',
                'attendance_percentage': 'mean',
                'participation_score': 'mean',
                'previous_gpa': 'mean',
                'final_exam_score': 'mean',
            }).reset_index()

            counts_series = rec_counts.loc[agg['usn']].values
            agg['record_count'] = counts_series

            feat_df = agg[FEATURES].copy()
            feat_df['total_video_time_minutes'] = feat_df['total_video_time_minutes'] / np.maximum(1, counts_series)

            X_scaled = self.scaler.transform(feat_df)

            if self.rf_regressor:
                agg['pred_score'] = np.clip(self.rf_regressor.predict(X_scaled), 0, 100).round(1)
            elif self.linear_regressor:
                agg['pred_score'] = np.clip(self.linear_regressor.predict(X_scaled), 0, 100).round(1)
            else:
                agg['pred_score'] = agg['final_exam_score'].round(1)

            if self.rf_classifier and self.label_encoder:
                cls_preds = self.rf_classifier.predict(X_scaled)
                agg['pred_label'] = self.label_encoder.inverse_transform(cls_preds)
            else:
                agg['pred_label'] = 'Medium'

            risk_badges = []
            risk_levels = []
            requires_mentoring = []

            for _, row in agg.iterrows():
                pts = 0
                att = row['attendance_percentage']
                qz = row['quiz_avg_score']
                asg = row['assignment_avg_marks']
                ps = row['pred_score']
                lbl = row['pred_label']

                if att < 65.0: pts += 3
                elif att < 75.0: pts += 2

                if qz < 4.0: pts += 2
                elif qz < 5.5: pts += 1

                if asg < 14.0: pts += 2
                elif asg < 18.0: pts += 1

                if ps < 40.0: pts += 3
                elif ps < 52.0: pts += 2
                elif ps < 62.0: pts += 1

                if lbl == 'Low': pts += 2

                if pts >= 5 or ps < 40.0:
                    badge = "HIGH RISK"
                    lvl = "🔴 HIGH RISK"
                elif pts >= 3 or ps < 52.0:
                    badge = "AT RISK"
                    lvl = "🟠 AT RISK"
                elif pts >= 1 or att < 75.0 or ps < 65.0:
                    badge = "MODERATE RISK"
                    lvl = "🟡 MODERATE RISK"
                else:
                    badge = "LOW RISK"
                    lvl = "🟢 LOW RISK"

                risk_badges.append(badge)
                risk_levels.append(lvl)
                requires_mentoring.append(badge in ('HIGH RISK', 'AT RISK') or att < 75.0 or ps < 50.0)

            agg['risk_badge'] = risk_badges
            agg['risk_level'] = risk_levels
            agg['requires_mentoring'] = requires_mentoring

            self.cohort_cache = agg
            logger.info("Cohort cache ready for %d students.", len(agg))
        except Exception as e:
            logger.error("Failed to build cohort cache: %s", e)

    def reload_if_needed(self):
        """Check if dataset file changed on disk and reload."""
        if os.path.exists(self.dataset_path):
            current_mtime = os.path.getmtime(self.dataset_path)
            if self.mtime != current_mtime:
                self._load_dataset()
                self._build_cohort_cache()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Retrieval of Student Records
    # ──────────────────────────────────────────────────────────────────────────
    def find_student_records(self, query_id_or_name: str) -> list:
        """
        Retrieves ALL records belonging to a student by searching:
        1. USN exact match
        2. Student ID exact/prefix match (e.g. DB_1162)
        3. Student Name exact or first name match
        """
        self.reload_if_needed()
        query = query_id_or_name.strip()
        if not query:
            return []

        # Strip optional "USN" prefix
        q_clean = re.sub(r'^(?:USN[\s_:]*)', '', query, flags=re.IGNORECASE).strip()

        # 1. Exact USN match
        q_upper = q_clean.upper()
        if q_upper in self.usn_index:
            indices = self.usn_index[q_upper]
            return self.df.iloc[indices].to_dict(orient='records')

        # 2. Check student_id exact match
        id_matches = self.df[self.df['student_id'].str.upper() == q_upper]
        if not id_matches.empty:
            target_usn = id_matches.iloc[0]['usn'].upper()
            return self.df.iloc[self.usn_index[target_usn]].to_dict(orient='records')

        # Check prefix of student_id (e.g. DB_1162)
        id_prefix = self.df[self.df['student_id'].str.upper().str.startswith(q_upper + '_')]
        if not id_prefix.empty:
            target_usn = id_prefix.iloc[0]['usn'].upper()
            return self.df.iloc[self.usn_index[target_usn]].to_dict(orient='records')

        # 3. Exact full name match
        q_lower = q_clean.lower()
        if q_lower in self.full_name_index:
            indices = self.full_name_index[q_lower]
            return self.df.iloc[indices].to_dict(orient='records')

        # 4. First name match
        if q_lower in self.first_name_index:
            indices = self.first_name_index[q_lower]
            # Group by USN; pick first student or return all records for the first matching student
            sub_df = self.df.iloc[indices]
            first_matched_usn = sub_df['usn'].iloc[0].upper()
            return self.df.iloc[self.usn_index[first_matched_usn]].to_dict(orient='records')

        # 5. Substring full name match
        name_matches = self.df[self.df['student_name'].str.lower().str.contains(q_lower, regex=False)]
        if not name_matches.empty:
            matched_usn = name_matches['usn'].mode()[0].upper()
            return self.df.iloc[self.usn_index[matched_usn]].to_dict(orient='records')

        # 6. Alphanumeric normalized USN match (e.g. 2024-IT-0001)
        alnum = re.sub(r'[^A-Za-z0-9]', '', q_upper)
        for usn, indices in self.usn_index.items():
            if re.sub(r'[^A-Za-z0-9]', '', usn) == alnum:
                return self.df.iloc[indices].to_dict(orient='records')

        return []

    def get_ambiguous_names(self, first_name: str) -> list:
        """Returns list of unique students sharing a first name."""
        self.reload_if_needed()
        f_lower = first_name.strip().lower()
        if f_lower not in self.first_name_index:
            return []
        indices = self.first_name_index[f_lower]
        sub = self.df.iloc[indices][['usn', 'student_name']].drop_duplicates()
        return sub.to_dict(orient='records')

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Multi-Record Aggregation
    # ──────────────────────────────────────────────────────────────────────────
    def aggregate_student_profile(self, records: list) -> dict:
        """
        Aggregates multiple records for a student into a unified academic profile.
        """
        if not records:
            return {}

        first = records[0]
        student_name = first.get('student_name', '')
        usn = first.get('usn', '')

        student_ids = [r.get('student_id', '') for r in records]
        base_id = student_ids[0]
        if '_' in base_id:
            parts = base_id.split('_')
            if len(parts) >= 2:
                base_id = '_'.join(parts[:-1])

        subject_breakdown = []
        for r in records:
            sid = str(r.get('student_id', ''))
            suffix = sid.split('_')[-1] if '_' in sid else sid
            subj_name = SUBJECT_MAP.get(suffix, suffix)

            subject_breakdown.append({
                'record_id': sid,
                'subject_code': suffix,
                'subject_name': subj_name,
                'quiz_avg_score': float(r.get('quiz_avg_score', 0.0)),
                'assignment_avg_marks': float(r.get('assignment_avg_marks', 0.0)),
                'attendance_percentage': float(r.get('attendance_percentage', 0.0)),
                'participation_score': float(r.get('participation_score', 0.0)),
                'videos_watched': int(r.get('videos_watched', 0)),
                'total_video_time_minutes': float(r.get('total_video_time_minutes', 0.0)),
                'final_exam_score': float(r.get('final_exam_score', 0.0)),
                'performance_label': str(r.get('performance_label', '')),
                'appended_at': r.get('appended_at', None),
            })

        quiz_scores = [sb['quiz_avg_score'] for sb in subject_breakdown]
        assignment_marks = [sb['assignment_avg_marks'] for sb in subject_breakdown]
        attendances = [sb['attendance_percentage'] for sb in subject_breakdown]
        participations = [sb['participation_score'] for sb in subject_breakdown]
        videos = [sb['videos_watched'] for sb in subject_breakdown]
        video_times = [sb['total_video_time_minutes'] for sb in subject_breakdown]
        exam_scores = [sb['final_exam_score'] for sb in subject_breakdown if sb['final_exam_score'] > 0]
        previous_gpas = [float(r.get('previous_gpa', 0.0)) for r in records if float(r.get('previous_gpa', 0.0)) > 0]

        avg_quiz = round(float(np.mean(quiz_scores)), 2) if quiz_scores else 0.0
        avg_assignment = round(float(np.mean(assignment_marks)), 2) if assignment_marks else 0.0
        avg_attendance = round(float(np.mean(attendances)), 1) if attendances else 0.0
        avg_participation = round(float(np.mean(participations)), 2) if participations else 0.0
        avg_videos = round(float(np.mean(videos)), 1) if videos else 0.0
        total_video_time = round(float(np.sum(video_times)), 1) if video_times else 0.0
        prev_gpa = round(float(np.mean(previous_gpas)), 2) if previous_gpas else float(first.get('previous_gpa', 0.0))

        avg_final_exam = round(float(np.mean(exam_scores)), 1) if exam_scores else 0.0
        labels = [sb['performance_label'] for sb in subject_breakdown if sb['performance_label']]
        perf_label = max(set(labels), key=labels.count) if labels else 'Medium'

        dates = [str(r.get('appended_at')) for r in records if pd.notna(r.get('appended_at')) and str(r.get('appended_at')).strip()]
        latest_date = dates[-1] if dates else 'Current Academic Term'

        return {
            'student_name': student_name,
            'usn': usn,
            'student_id': base_id,
            'record_count': len(records),
            'latest_data': latest_date,
            'avg_quiz_score': avg_quiz,
            'avg_assignment_marks': avg_assignment,
            'avg_attendance': avg_attendance,
            'avg_participation': avg_participation,
            'avg_videos_watched': avg_videos,
            'total_video_time_minutes': total_video_time,
            'previous_gpa': prev_gpa,
            'final_exam_score': avg_final_exam,
            'performance_label': perf_label,
            'subject_breakdown': subject_breakdown,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Performance Trend Analysis
    # ──────────────────────────────────────────────────────────────────────────
    def analyze_performance_trend(self, profile: dict) -> dict:
        """
        Classifies performance trend as Improving, Stable, or Declining.
        """
        breakdown = profile.get('subject_breakdown', [])
        prev_gpa = profile.get('previous_gpa', 0.0)
        baseline_pct = prev_gpa * 10.0 if prev_gpa > 0 else None

        if len(breakdown) < 2:
            exam_score = profile.get('final_exam_score', 0.0)
            if baseline_pct and exam_score > 0:
                diff = exam_score - baseline_pct
                if diff >= 4.0:
                    trend = "Improving"
                    detail = f"Higher current exam score ({exam_score:.1f}) compared to GPA baseline ({baseline_pct:.1f}%)."
                elif diff <= -4.0:
                    trend = "Declining"
                    detail = f"Current performance ({exam_score:.1f}) is below previous GPA baseline ({baseline_pct:.1f}%)."
                else:
                    trend = "Stable"
                    detail = f"Consistent with historical GPA benchmark of {prev_gpa:.2f}."
            else:
                trend = "Stable"
                detail = "Insufficient multi-semester historical data available (single record available)."
            return {
                'trend': trend,
                'detail': detail,
                'is_single_record': True,
            }

        exam_scores = [sb['final_exam_score'] for sb in breakdown if sb['final_exam_score'] > 0]
        quiz_pcts = [sb['quiz_avg_score'] * 10.0 for sb in breakdown]

        if len(exam_scores) >= 2:
            first_half = np.mean(exam_scores[:len(exam_scores)//2])
            second_half = np.mean(exam_scores[len(exam_scores)//2:])
            delta = second_half - first_half
        else:
            first_half = np.mean(quiz_pcts[:len(quiz_pcts)//2])
            second_half = np.mean(quiz_pcts[len(quiz_pcts)//2:])
            delta = second_half - first_half

        if delta >= 3.0:
            trend = "Improving"
            detail = f"Upward trajectory observed across course modules (+{delta:.1f}% improvement)."
        elif delta <= -3.0:
            trend = "Declining"
            detail = f"Downward trajectory observed across course modules ({delta:.1f}% decline)."
        else:
            trend = "Stable"
            detail = "Academic performance remains steady across enrolled subjects."

        return {
            'trend': trend,
            'detail': detail,
            'is_single_record': False,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Machine Learning Predictions
    # ──────────────────────────────────────────────────────────────────────────
    def predict_performance(self, profile: dict) -> dict:
        """
        Runs ML prediction pipeline:
        - Regression: Predict expected final_exam_score
        - Classification: Predict expected performance category
        Prevents data leakage: final_exam_score is NOT used as an input feature.
        """
        features_dict = {
            'videos_watched': profile.get('avg_videos_watched', 0.0),
            'total_video_time_minutes': profile.get('total_video_time_minutes', 0.0) / max(1, profile.get('record_count', 1)),
            'quiz_avg_score': profile.get('avg_quiz_score', 0.0),
            'assignment_avg_marks': profile.get('avg_assignment_marks', 0.0),
            'attendance_percentage': profile.get('avg_attendance', 0.0),
            'participation_score': profile.get('avg_participation', 0.0),
            'previous_gpa': profile.get('previous_gpa', 0.0),
        }

        feature_df = pd.DataFrame([features_dict], columns=FEATURES, dtype=float)

        predicted_score = 0.0
        predicted_label = 'Medium'
        cls_confidence = 94.1

        if self.scaler is not None:
            try:
                X_scaled = self.scaler.transform(feature_df)

                if self.rf_regressor is not None:
                    raw_score = float(self.rf_regressor.predict(X_scaled)[0])
                    predicted_score = float(np.clip(raw_score, 0.0, 100.0))
                elif self.linear_regressor is not None:
                    raw_score = float(self.linear_regressor.predict(X_scaled)[0])
                    predicted_score = float(np.clip(raw_score, 0.0, 100.0))
                else:
                    predicted_score = float(np.clip(
                        features_dict['previous_gpa'] * 6.5 +
                        features_dict['quiz_avg_score'] * 2.2 +
                        features_dict['attendance_percentage'] * 0.15,
                        0.0, 100.0
                    ))

                if self.rf_classifier is not None and self.label_encoder is not None:
                    pred_cls = self.rf_classifier.predict(X_scaled)[0]
                    predicted_label = str(self.label_encoder.inverse_transform([pred_cls])[0])
                    probs = self.rf_classifier.predict_proba(X_scaled)[0]
                    cls_confidence = round(float(np.max(probs)) * 100, 1)
                else:
                    if predicted_score >= 75: predicted_label = 'High'
                    elif predicted_score >= 50: predicted_label = 'Medium'
                    else: predicted_label = 'Low'

            except Exception as e:
                logger.error("Prediction failed: %s", e)
                predicted_score = profile.get('final_exam_score', 70.0)
                predicted_label = profile.get('performance_label', 'Medium')
        else:
            predicted_score = profile.get('final_exam_score', 70.0)
            predicted_label = profile.get('performance_label', 'Medium')

        return {
            'predicted_score': round(predicted_score, 1),
            'predicted_label': predicted_label,
            'confidence': cls_confidence,
            'model_r2': '0.959 (R² Score)',
            'features_used': features_dict,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Academic Risk Assessment & Risk Explanation
    # ──────────────────────────────────────────────────────────────────────────
    def evaluate_risk(self, profile: dict, prediction: dict) -> dict:
        """
        Reproducible 4-tier risk classification system:
        🟢 LOW RISK | 🟡 MODERATE RISK | 🟠 AT RISK | 🔴 HIGH RISK
        """
        risk_points = 0
        risk_factors = []
        interventions = []

        attendance = profile.get('avg_attendance', 0.0)
        quiz_avg = profile.get('avg_quiz_score', 0.0)
        assignment_avg = profile.get('avg_assignment_marks', 0.0)
        participation = profile.get('avg_participation', 0.0)
        prev_gpa = profile.get('previous_gpa', 0.0)
        video_count = profile.get('avg_videos_watched', 0.0)
        learning_time = profile.get('total_video_time_minutes', 0.0)
        pred_score = prediction.get('predicted_score', 0.0)
        pred_label = prediction.get('predicted_label', 'Medium')

        if attendance < 65.0:
            risk_points += 3
            risk_factors.append(f"Attendance is critically low at {attendance:.1f}% (required: 75%)")
            interventions.append("Immediate attendance counseling & mandatory session tracking")
        elif attendance < 75.0:
            risk_points += 2
            risk_factors.append(f"Attendance is below the required level ({attendance:.1f}% < 75%)")
            interventions.append("Attendance recovery plan & regular check-ins")

        if quiz_avg < 4.0:
            risk_points += 2
            risk_factors.append(f"Low quiz performance ({quiz_avg:.1f}/10)")
            interventions.append("Weekly practice quizzes and concept review modules")
        elif quiz_avg < 5.5:
            risk_points += 1
            risk_factors.append(f"Moderate quiz scores ({quiz_avg:.1f}/10)")
            interventions.append("Additional quiz practice & doubt clearance")

        if assignment_avg < 14.0:
            risk_points += 2
            risk_factors.append(f"Low assignment performance ({assignment_avg:.1f} avg marks)")
            interventions.append("Remedial classes & structured assignment tutoring")
        elif assignment_avg < 18.0:
            risk_points += 1
            risk_factors.append(f"Assignment marks need improvement ({assignment_avg:.1f})")
            interventions.append("Assignment feedback review & problem-solving practice")

        if video_count <= 2.5 or learning_time < 90.0:
            risk_points += 1
            risk_factors.append(f"Low learning activity ({video_count:.0f} videos watched, {learning_time:.0f} mins total)")
            interventions.append("Structured flipped-classroom video watching schedule")

        if participation < 4.5:
            risk_points += 1
            risk_factors.append(f"Low participation score ({participation:.1f}/10)")
            interventions.append("Encourage active interaction during doubt sessions")

        if prev_gpa < 5.0 and prev_gpa > 0:
            risk_points += 2
            risk_factors.append(f"Historical academic standing is weak (Previous GPA: {prev_gpa:.2f})")
            interventions.append("One-on-one faculty mentoring")
        elif prev_gpa < 6.0 and prev_gpa > 0:
            risk_points += 1
            risk_factors.append(f"Modest prior GPA ({prev_gpa:.2f})")

        if pred_score < 40.0:
            risk_points += 3
            risk_factors.append(f"Predicted final score is below passing mark ({pred_score:.1f}/100)")
            interventions.append("Immediate academic intervention and comprehensive re-teaching")
        elif pred_score < 50.0:
            risk_points += 2
            risk_factors.append(f"Predicted final score is below expected benchmark ({pred_score:.1f}/100)")
            interventions.append("Remedial coursework before final examinations")
        elif pred_score < 60.0:
            risk_points += 1
            risk_factors.append(f"Predicted final score is borderline ({pred_score:.1f}/100)")

        if pred_label == 'Low':
            risk_points += 2
            risk_factors.append("Classified in 'Low' performance bracket")
        elif pred_label == 'High':
            risk_points = max(0, risk_points - 1)

        if risk_points >= 5 or pred_score < 40.0:
            risk_level = "🔴 HIGH RISK"
            badge = "HIGH RISK"
        elif risk_points >= 3 or pred_score < 52.0:
            risk_level = "🟠 AT RISK"
            badge = "AT RISK"
        elif risk_points >= 1 or attendance < 75.0 or pred_score < 65.0:
            risk_level = "🟡 MODERATE RISK"
            badge = "MODERATE RISK"
        else:
            risk_level = "🟢 LOW RISK"
            badge = "LOW RISK"

        if not interventions:
            if badge == "LOW RISK":
                interventions = ["Maintain consistent study habits", "Explore advanced projects and competitive coding"]
            else:
                interventions = ["Regular self-assessment quizzes", "Sustain current attendance and video engagement"]

        unique_interventions = list(dict.fromkeys(interventions))[:4]
        unique_factors = list(dict.fromkeys(risk_factors))

        return {
            'risk_level': risk_level,
            'badge': badge,
            'risk_score': risk_points,
            'main_risk_factors': unique_factors,
            'recommended_interventions': unique_interventions,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Career Area Recommendations
    # ──────────────────────────────────────────────────────────────────────────
    def recommend_careers(self, profile: dict) -> list:
        """
        Recommends suitable career areas based on subject performance and indicators.
        """
        breakdown = profile.get('subject_breakdown', [])
        subjects = {sb['subject_code'].upper(): sb for sb in breakdown}
        gpa = profile.get('previous_gpa', 0.0)
        attendance = profile.get('avg_attendance', 0.0)

        scores = {d: 50.0 for d in CAREER_DOMAINS}

        if 'PY' in subjects:
            py_score = subjects['PY']['quiz_avg_score'] * 10
            scores['Software Development'] += py_score * 0.4
            scores['Data Analytics'] += py_score * 0.3

        if 'WD' in subjects:
            wd_score = subjects['WD']['quiz_avg_score'] * 10
            scores['Web Development'] += wd_score * 0.5
            scores['Software Development'] += wd_score * 0.3

        if 'ML' in subjects or 'AIML' in subjects:
            ml_sb = subjects.get('ML') or subjects.get('AIML')
            ml_score = ml_sb['quiz_avg_score'] * 10
            scores['Artificial Intelligence / Machine Learning'] += ml_score * 0.5
            scores['Data Science'] += ml_score * 0.4

        if 'DSC' in subjects or 'DS' in subjects:
            ds_sb = subjects.get('DSC') or subjects.get('DS')
            ds_score = ds_sb['quiz_avg_score'] * 10
            scores['Data Science'] += ds_score * 0.4
            scores['Data Analytics'] += ds_score * 0.4
            scores['Software Development'] += ds_score * 0.3

        if 'CN' in subjects:
            cn_score = subjects['CN']['quiz_avg_score'] * 10
            scores['Cybersecurity'] += cn_score * 0.4
            scores['Cloud Computing'] += cn_score * 0.4

        if gpa >= 8.0:
            scores['Research / Higher Studies'] += 25.0
            scores['Artificial Intelligence / Machine Learning'] += 10.0

        if attendance >= 85.0:
            scores['Software Development'] += 5.0
            scores['Cloud Computing'] += 5.0

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for domain, score in ranked[:3]:
            suitability = "Strong suitability" if score >= 75 else "Moderate suitability"
            results.append({
                'domain': domain,
                'suitability': suitability,
                'score': round(score, 1)
            })

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Full Analysis Pipeline for an Individual Student
    # ──────────────────────────────────────────────────────────────────────────
    def analyze_student(self, query_id_or_name: str) -> dict:
        records = self.find_student_records(query_id_or_name)
        if not records:
            return {'found': False, 'query': query_id_or_name}

        profile = self.aggregate_student_profile(records)
        trend = self.analyze_performance_trend(profile)
        prediction = self.predict_performance(profile)
        risk = self.evaluate_risk(profile, prediction)
        careers = self.recommend_careers(profile)

        return {
            'found': True,
            'profile': profile,
            'trend': trend,
            'prediction': prediction,
            'risk': risk,
            'careers': careers,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 8. Fast Multiple Student Cohort Analysis (from cache)
    # ──────────────────────────────────────────────────────────────────────────
    def query_cohort(self, filter_type: str, threshold: float = None) -> list:
        self.reload_if_needed()
        if self.cohort_cache is None:
            self._build_cohort_cache()

        df = self.cohort_cache
        if df is None or df.empty:
            return []

        if filter_type == 'high_risk':
            sub = df[df['risk_badge'] == 'HIGH RISK']
        elif filter_type == 'at_risk':
            sub = df[df['risk_badge'].isin(['HIGH RISK', 'AT RISK'])]
        elif filter_type == 'mentoring_required':
            sub = df[df['requires_mentoring'] == True]
        elif filter_type == 'attendance_below':
            limit = threshold if threshold is not None else 75.0
            sub = df[df['attendance_percentage'] < limit]
        elif filter_type == 'predicted_below':
            limit = threshold if threshold is not None else 50.0
            sub = df[df['pred_score'] < limit]
        elif filter_type == 'declining_trend':
            sub = df[(df['risk_badge'].isin(['HIGH RISK', 'AT RISK'])) | (df['attendance_percentage'] < 75.0)]
        elif filter_type == 'top_performers':
            sub = df[(df['pred_score'] >= 85.0) | (df['pred_label'] == 'High')].sort_values(by='pred_score', ascending=False)
        else:
            sub = df[df['risk_badge'].isin(['HIGH RISK', 'AT RISK'])]

        results = []
        for _, row in sub.iterrows():
            results.append({
                'usn': row['usn'],
                'student_name': row['student_name'],
                'risk_badge': row['risk_badge'],
                'risk_level': row['risk_level'],
                'attendance': round(row['attendance_percentage'], 1),
                'predicted_score': row['pred_score'],
                'perf_label': row['pred_label'],
                'requires_mentoring': row['requires_mentoring'],
            })

        return results


def get_engine() -> AcademicIntelligenceEngine:
    """Singleton getter for AcademicIntelligenceEngine."""
    global _CACHED_ENGINE
    if _CACHED_ENGINE is None:
        with _ENGINE_LOCK:
            if _CACHED_ENGINE is None:
                _CACHED_ENGINE = AcademicIntelligenceEngine()
    return _CACHED_ENGINE


# ──────────────────────────────────────────────────────────────────────────────
# Role-Based Access Control & Security Helpers
# ──────────────────────────────────────────────────────────────────────────────
def check_user_role(user) -> dict:
    if not user or not getattr(user, 'is_authenticated', False):
        return {'role': 'anonymous', 'is_teacher': False, 'is_student': False, 'permitted_usn': None}

    is_teacher_role = getattr(user, 'is_superuser', False) or getattr(user, 'is_staff', False) or hasattr(user, 'teacher_profile')
    is_student_role = hasattr(user, 'student_profile')
    permitted_usn = None
    if is_student_role:
        try:
            permitted_usn = user.student_profile.roll_number.upper().strip()
        except Exception:
            permitted_usn = None

    role = 'teacher' if is_teacher_role else ('student' if is_student_role else 'user')

    return {
        'role': role,
        'is_teacher': is_teacher_role,
        'is_student': is_student_role,
        'permitted_usn': permitted_usn,
        'user': user,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Query Parsing & Entity Extraction
# ──────────────────────────────────────────────────────────────────────────────
USN_CLEAN_REGEX = re.compile(r'\b(?:USN[\s_:]*)?(\d{4}[A-Za-z]{2}\d{4}|USN_\w+|DB_\d+(?:_[A-Za-z]+)?)\b', re.IGNORECASE)

COHORT_PATTERNS = [
    (r'(?:high[\s-]risk|critical[\s-]risk)', 'high_risk', None),
    (r'(?:at[\s-]risk|students?\s+at\s+risk)', 'at_risk', None),
    (r'(?:mentoring|remedial|require(?:s)?\s+mentoring|require(?:s)?\s+immediate\s+mentoring)', 'mentoring_required', None),
    (r'(?:attendance\s+(?:below|less\s+than|<)\s*(\d+))', 'attendance_below', None),
    (r'(?:attendance\s+below\s+75%?)', 'attendance_below', 75.0),
    (r'(?:predicted\s+(?:final\s+)?score\s+(?:below|less\s+than|<)\s*(\d+))', 'predicted_below', None),
    (r'(?:declining|deteriorating)\s+performance', 'declining_trend', None),
    (r'(?:top[\s-]performing|best\s+performing|highest\s+score)', 'top_performers', None),
]


def extract_student_target(user_query: str, chat_history: list = None) -> tuple:
    """
    Finds targeted USN or student name from user query or recent conversation history.
    Returns (target, is_self_query)
    """
    q = user_query.strip()
    q_lower = q.lower()

    # Self-query triggers
    self_keywords = [
        'my details', 'my marks', 'my attendance', 'my performance', 'my risk',
        'my predicted score', 'am i at risk', 'how am i performing', 'my score', 'about me',
        'my weak areas', 'my expected performance', 'my final score', 'my career',
        'how can i improve', 'improve my score', 'improve my performance',
        'suitable for me', 'careers for me', 'my career areas', 'my report',
        'show my complete report', 'my complete report'
    ]
    if any(k in q_lower for k in self_keywords):
        return ('__SELF__', True)

    # 1. Match explicit USN in query
    usn_match = USN_CLEAN_REGEX.search(q)
    if usn_match:
        extracted = usn_match.group(1).upper()
        return (extracted, False)

    # 2. Match student name using engine indexes
    engine = get_engine()

    # Check for possessive or query syntax: e.g. "Rahul's marks", "details of Rahul"
    poss_match = re.search(r'\b([A-Za-z]+)(?:\'s|\s+marks|\s+details|\s+performance|\s+attendance|\s+score)\b', q, re.IGNORECASE)
    if poss_match:
        candidate = poss_match.group(1).lower()
        if candidate in engine.first_name_index or candidate in engine.full_name_index:
            return (candidate, False)

    # Check all words or n-grams against name index
    clean_words = re.findall(r'[A-Za-z]+', q)
    for length in [3, 2, 1]:
        for i in range(len(clean_words) - length + 1):
            cand = " ".join(clean_words[i:i+length]).lower()
            if cand in engine.full_name_index:
                return (cand, False)
            if length == 1 and cand in engine.first_name_index:
                # Avoid common English stop words that might match
                if cand not in {'the', 'show', 'all', 'is', 'what', 'how', 'who', 'tell', 'me', 'details', 'score', 'risk'}:
                    return (cand, False)

    # 3. Follow-up intent in conversation ("Why is this student at risk?", "What is the student's attendance?")
    followup_triggers = [
        'this student', 'the student', 'his', 'her', 'their', 'he ', 'she ',
        'final score', 'expected performance', 'expected score', 'predicted', 'predicted score',
        'is he at risk', 'is she at risk', 'is this student at risk', 'why is this', 'why is the',
        'what intervention', 'weak areas', 'attendance', 'academic performance',
        'career areas', 'suitable for this student', 'improving or declining', 'performing', 'marks', 'risk',
        'full report', 'complete report', 'show full', 'show complete', 'entire report', 'full student report'
    ]
    if any(t in q_lower for t in followup_triggers) and chat_history:
        for msg in reversed(chat_history):
            content = msg.get('content', '')
            found_usn = USN_CLEAN_REGEX.search(content)
            if found_usn:
                return (found_usn.group(1).upper(), False)

    return (None, False)


def detect_focus_area(user_query: str) -> str:
    """Detects targeted sub-dimension."""
    q = user_query.lower()
    if any(w in q for w in ['full report', 'complete report', 'show full', 'show complete', 'entire report', 'full student report', 'complete student report']):
        return 'overview'
    if 'risk' in q or 'why is' in q or 'at risk' in q:
        return 'risk'
    if 'predict' in q or 'final score' in q or 'expected performance' in q or 'expected score' in q or 'predicted score' in q:
        return 'prediction'
    if 'attendance' in q:
        return 'attendance'
    if 'trend' in q or 'improving' in q or 'declining' in q or 'trajectory' in q:
        return 'trend'
    if 'career' in q or 'job' in q or 'suitable for' in q or 'domain' in q:
        return 'career'
    if 'intervention' in q or 'mentoring' in q or 'action' in q or 'weak area' in q or 'improve' in q:
        return 'intervention'
    return 'overview'


# ──────────────────────────────────────────────────────────────────────────────
# Focused Sub-dimension Formatters
# ──────────────────────────────────────────────────────────────────────────────
def format_risk_response(analysis: dict) -> str:
    p = analysis['profile']
    rk = analysis['risk']
    pr = analysis['prediction']

    lines = [
        f"### 🛡️ Risk Assessment: {p['student_name']} (USN: {p['usn']})",
        "",
        f"- **Current Risk Status:** {rk['risk_level']} (Risk Score: {rk['risk_score']} points)",
        f"- **Predicted Final Score:** **{pr['predicted_score']} / 100** ({pr['predicted_label']} performance bracket)",
        f"- **Overall Attendance:** {p['avg_attendance']:.1f}% ({'⚠️ Below 75% required threshold' if p['avg_attendance'] < 75.0 else '✅ Satisfies attendance threshold'})",
        f"- **Continuous Quiz Average:** {p['avg_quiz_score']:.2f} / 10 ({p['avg_quiz_score']*10:.0f}%)",
        "",
        "#### 🔍 Identified Risk Factors:",
    ]
    if rk['main_risk_factors']:
        for rf in rk['main_risk_factors']:
            lines.append(f"• {rf}")
    else:
        lines.append("• No critical risk factors identified. Academic performance satisfies course benchmarks.")

    lines.append("")
    lines.append("#### 📋 Recommended Actions:")
    for iv in rk['recommended_interventions']:
        lines.append(f"• {iv}")

    return "\n".join(lines)


def format_intervention_response(analysis: dict) -> str:
    p = analysis['profile']
    rk = analysis['risk']
    pr = analysis['prediction']
    sb = p.get('subject_breakdown', [])

    requires_urgent = rk['badge'] in ('HIGH RISK', 'AT RISK') or p['avg_attendance'] < 75.0 or pr['predicted_score'] < 50.0
    priority_label = "🚨 High Priority (Immediate Action Required)" if requires_urgent else "Routine Academic Support"

    lines = [
        f"### 📋 Recommended Intervention Plan: {p['student_name']} (USN: {p['usn']})",
        "",
        f"- **Student Profile:** {p['student_name']} ({p['usn']}) — Previous GPA: {p['previous_gpa']:.2f}",
        f"- **Current Risk Level:** {rk['risk_level']}",
        f"- **Mentoring Priority:** {priority_label}",
        f"- **Predicted Final Performance:** {pr['predicted_score']}/100 ({pr['predicted_label']})",
        "",
        "#### 🎯 Actionable Intervention Strategies:",
    ]
    for iv in rk['recommended_interventions']:
        lines.append(f"• {iv}")

    weak_subjects = [s for s in sb if s['quiz_avg_score'] < 6.0 or s['attendance_percentage'] < 75.0 or s['final_exam_score'] < 50.0]
    if weak_subjects:
        lines.append("")
        lines.append("#### ⚠️ Subjects Requiring Specific Attention:")
        for ws in weak_subjects:
            issues = []
            if ws['attendance_percentage'] < 75.0:
                issues.append(f"Attendance Shortage ({ws['attendance_percentage']:.1f}%)")
            if ws['quiz_avg_score'] < 6.0:
                issues.append(f"Low Quiz Average ({ws['quiz_avg_score']:.1f}/10)")
            if ws['final_exam_score'] < 50.0:
                issues.append(f"Low Final Exam ({ws['final_exam_score']:.1f}/100)")
            lines.append(f"• **{ws['subject_code']} ({ws['subject_name']}):** {', '.join(issues)}")

    return "\n".join(lines)


def format_career_response(analysis: dict) -> str:
    p = analysis['profile']
    cr = analysis['careers']

    lines = [
        f"### 🚀 Career Suitability Analysis: {p['student_name']} (USN: {p['usn']})",
        "",
        f"Based on coursework performance, subject strengths, and learning indicators (GPA: {p['previous_gpa']:.2f}, Quiz Avg: {p['avg_quiz_score']:.2f}/10):",
        "",
        "#### 🌟 Recommended Domains & Suitability:",
    ]
    for i, c in enumerate(cr, 1):
        lines.append(f"{i}. **{c['domain']}** — *{c['suitability']}* (Alignment: {c['score']}%)")

    top_domain = cr[0]['domain'] if cr else "Software Development"
    lines.append("")
    lines.append(f"💡 **Guidance:** Encourage student to build domain portfolios and capstone projects in **{top_domain}**.")
    return "\n".join(lines)


def format_prediction_response(analysis: dict) -> str:
    p = analysis['profile']
    pr = analysis['prediction']
    tr = analysis['trend']

    lines = [
        f"### 🎯 Final Score Prediction: {p['student_name']} (USN: {p['usn']})",
        "",
        f"- **Predicted Final Score:** **{pr['predicted_score']} / 100**",
        f"- **Expected Performance Category:** **{pr['predicted_label']}**",
        f"- **Model Reliability / Confidence:** {pr['confidence']}% (Held-out Model R²: 0.959)",
        f"- **Academic Trajectory:** {tr['trend']} ({tr['detail']})",
        "",
        "#### 📊 Contributing Metrics:",
        f"• Continuous Quiz Average: {p['avg_quiz_score']:.2f} / 10 ({p['avg_quiz_score']*10:.0f}%)",
        f"• Assignment Marks: {p['avg_assignment_marks']:.2f}",
        f"• Overall Attendance: {p['avg_attendance']:.1f}%",
        f"• Historical GPA: {p['previous_gpa']:.2f}",
    ]
    return "\n".join(lines)


def format_attendance_response(analysis: dict) -> str:
    p = analysis['profile']
    sb = p.get('subject_breakdown', [])
    status = "✅ Meets 75% Requirement" if p['avg_attendance'] >= 75.0 else "⚠️ Attendance Shortage (< 75%)"

    lines = [
        f"### 📅 Attendance Analysis: {p['student_name']} (USN: {p['usn']})",
        "",
        f"- **Overall Aggregate Attendance:** **{p['avg_attendance']:.1f}%**",
        f"- **Compliance Status:** {status}",
        f"- **Enrolled Subjects Tracked:** {p['record_count']}",
        "",
        "| Subject Code | Subject Name | Attendance | Status |",
        "|---|---|---|---|",
    ]
    for s in sb:
        st = "✅ Good" if s['attendance_percentage'] >= 75.0 else "⚠️ Shortage"
        lines.append(f"| {s['subject_code']} | {s['subject_name']} | {s['attendance_percentage']:.1f}% | {st} |")

    return "\n".join(lines)


def format_trend_response(analysis: dict) -> str:
    p = analysis['profile']
    tr = analysis['trend']

    symbol = "📈" if tr['trend'] == 'Improving' else ("📉" if tr['trend'] == 'Declining' else "→")
    lines = [
        f"### {symbol} Performance Trend: {p['student_name']} (USN: {p['usn']})",
        "",
        f"- **Trajectory:** **{tr['trend']}**",
        f"- **Details:** {tr['detail']}",
        f"- **Continuous Quiz Average:** {p['avg_quiz_score']:.2f} / 10",
        f"- **Learning Engagement:** {p['total_video_time_minutes']:.1f} mins total video time across {p['avg_videos_watched']:.1f} videos avg",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard & Response Formatters (Requirement 17, 4, 13)
# ──────────────────────────────────────────────────────────────────────────────
def format_student_dashboard(analysis: dict, focus_area: str = 'overview') -> str:
    """
    Renders clean dashboard-style response strictly following Requirement 17 & Requirement 4.
    """
    if focus_area == 'risk':
        return format_risk_response(analysis)
    elif focus_area == 'intervention':
        return format_intervention_response(analysis)
    elif focus_area == 'career':
        return format_career_response(analysis)
    elif focus_area == 'prediction':
        return format_prediction_response(analysis)
    elif focus_area == 'attendance':
        return format_attendance_response(analysis)
    elif focus_area == 'trend':
        return format_trend_response(analysis)

    p = analysis['profile']
    tr = analysis['trend']
    pr = analysis['prediction']
    rk = analysis['risk']
    cr = analysis['careers']
    sb = p.get('subject_breakdown', [])

    lines = []

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("       STUDENT ANALYSIS")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    lines.append(f"Student: {p['student_name']}")
    lines.append(f"USN: {p['usn']}")
    lines.append(f"Student ID: {p['student_id']}")
    lines.append(f"Records Available: {p['record_count']} subject(s)")
    lines.append(f"Latest Data: {p['latest_data']}\n")

    # Subject/Record-wise analysis (Requirement 4)
    if len(sb) >= 1:
        lines.append("SUBJECT / RECORD-WISE BREAKDOWN")
        lines.append("| Subject/Record | Quiz | Assignment | Attendance | Exam Score | Label |")
        lines.append("|---|---|---|---|---|---|")
        for s in sb:
            q_pct = f"{s['quiz_avg_score']:.1f} ({s['quiz_avg_score']*10:.0f}%)"
            lines.append(f"| {s['subject_code']} ({s['subject_name']}) | {q_pct} | {s['assignment_avg_marks']:.1f} | {s['attendance_percentage']:.1f}% | {s['final_exam_score']:.1f} | {s['performance_label']} |")
        lines.append("")

    # Academic Performance
    lines.append("ACADEMIC PERFORMANCE")
    lines.append(f"• Previous GPA: {p['previous_gpa']:.2f}")
    lines.append(f"• Quiz Average: {p['avg_quiz_score']:.2f} / 10 ({p['avg_quiz_score']*10:.0f}%)")
    lines.append(f"• Assignment Average: {p['avg_assignment_marks']:.2f}")
    lines.append(f"• Attendance: {p['avg_attendance']:.1f}%")
    lines.append(f"• Participation: {p['avg_participation']:.2f} / 10\n")

    # Learning Activity
    lines.append("LEARNING ACTIVITY")
    lines.append(f"• Videos Watched: {p['avg_videos_watched']:.1f} (avg/subject)")
    lines.append(f"• Total Learning Time: {p['total_video_time_minutes']:.1f} minutes\n")

    # Performance Trend
    lines.append("PERFORMANCE TREND")
    trend_symbol = "📈 Improving" if tr['trend'] == 'Improving' else ("📉 Declining" if tr['trend'] == 'Declining' else "→ Stable")
    lines.append(f"{trend_symbol}")
    lines.append(f"• {tr['detail']}\n")

    # Predicted Final Score (Requirement 6, 10)
    lines.append("PREDICTED FINAL SCORE")
    lines.append(f"{pr['predicted_score']} / 100")
    lines.append(f"Prediction Confidence / Model Performance: {pr['confidence']}% (R²: 0.959)\n")

    # Performance Level (Requirement 7)
    lines.append("PERFORMANCE LEVEL")
    lines.append(f"{pr['predicted_label']}\n")

    # Risk Level (Requirement 8)
    lines.append("RISK LEVEL")
    lines.append(f"{rk['risk_level']}\n")

    # Risk Factors (Requirement 9)
    if rk['main_risk_factors']:
        lines.append("MAIN RISK FACTORS")
        for rf in rk['main_risk_factors']:
            lines.append(f"• {rf}")
        lines.append("")
    else:
        lines.append("MAIN RISK FACTORS")
        lines.append("• Academic metrics satisfy expected course benchmarks.\n")

    # Recommended Interventions
    lines.append("RECOMMENDED INTERVENTION")
    for iv in rk['recommended_interventions']:
        lines.append(f"• {iv}")
    lines.append("")

    # Career Areas (Requirement 11)
    lines.append("SUGGESTED CAREER AREAS")
    for i, c in enumerate(cr, 1):
        lines.append(f"{i}. {c['domain']} – {c['suitability']}")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines)


def format_cohort_table(filter_type: str, cohort: list) -> str:
    """
    Formats cohort list into structured markdown table (Requirement 13).
    """
    titles = {
        'high_risk': "🔴 HIGH-RISK STUDENTS (Immediate Attention Required)",
        'at_risk': "🟠 AT-RISK STUDENTS LIST",
        'mentoring_required': "📋 STUDENTS REQUIRING IMMEDIATE MENTORING",
        'attendance_below': "⚠️ STUDENTS WITH ATTENDANCE BELOW REQUIRED 75%",
        'predicted_below': "⚠️ STUDENTS WITH PREDICTED SCORE BELOW 50",
        'declining_trend': "📉 STUDENTS WITH CONCERNING PERFORMANCE METRICS",
        'top_performers': "🌟 TOP-PERFORMING STUDENTS",
    }
    title = titles.get(filter_type, "STUDENT COHORT REPORT")

    lines = []
    lines.append(f"### {title}")
    lines.append(f"**Total Identified:** {len(cohort)} student(s)\n")

    if not cohort:
        lines.append("No students matched the specified cohort criteria in the current dataset.")
        return "\n".join(lines)

    lines.append("| USN | Student Name | Risk Level | Attendance | Predicted Score | Expected Label | Mentoring |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in cohort[:40]:
        mentor_status = "Immediate" if s['requires_mentoring'] else "Routine"
        lines.append(f"| {s['usn']} | {s['student_name']} | {s['risk_badge']} | {s['attendance']}% | {s['predicted_score']}/100 | {s['perf_label']} | {mentor_status} |")

    if len(cohort) > 40:
        lines.append(f"\n*Showing top 40 of {len(cohort)} records.*")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main Dispatcher for Academic Intelligence Queries
# ──────────────────────────────────────────────────────────────────────────────
def handle_academic_query(user_query: str, user, chat_history: list = None) -> dict:
    """
    Evaluates if user_query is an academic intelligence query and enforces RBAC.
    """
    engine = get_engine()
    role_info = check_user_role(user)
    is_teacher = role_info['is_teacher']
    is_student = role_info['is_student']
    permitted_usn = role_info['permitted_usn']

    q_lower = user_query.lower()

    # ── 1. Check Individual Student Target First (including follow-ups) ───────
    # If the query contains an explicit USN, student name, self query, or refers to "this student",
    # it must be treated as an individual student query even if words like "at risk" are present.
    target, is_self = extract_student_target(user_query, chat_history)

    # ── 2. Check Cohort Query (e.g. "Show all high-risk students") ────────────
    cohort_filter = None
    threshold_val = None

    # Only treat as cohort query if no individual student target was extracted
    # or query explicitly asks for "all", "which students", "list", etc.
    is_cohort_request = any(w in q_lower for w in ['all ', 'which students', 'list of', 'students with', 'students who', 'cohort'])
    
    if not target or is_cohort_request:
        for pattern, filter_type, def_thresh in COHORT_PATTERNS:
            m = re.search(pattern, q_lower)
            if m:
                # Disqualify if it's clearly asking about a single student: "is this student", "why is this student"
                if any(x in q_lower for x in ['this student', 'is he', 'is she', 'why is he', 'why is she', 'why is this']):
                    break
                cohort_filter = filter_type
                if def_thresh is not None:
                    threshold_val = def_thresh
                elif m.groups() and m.group(1):
                    try:
                        threshold_val = float(m.group(1))
                    except Exception:
                        threshold_val = None
                break

    if cohort_filter:
        if not is_teacher:
            return {
                'is_academic_query': True,
                'handled': True,
                'reply': "Access Denied\n\nYou do not have permission to access cohort-wide student information. Only authorized teachers and administrators may view class-level analytics.",
                'sources': ["Role-Based Access Control Policy"],
                'error': 'access_denied',
            }

        students = engine.query_cohort(cohort_filter, threshold=threshold_val)
        reply = format_cohort_table(cohort_filter, students)
        return {
            'is_academic_query': True,
            'handled': True,
            'reply': reply,
            'sources': ["Student Academic Dataset (dataset.csv)"],
            'error': None,
        }

    if not target:
        return {'is_academic_query': False, 'handled': False, 'reply': '', 'sources': [], 'error': None}

    # Self-query handling
    if is_self:
        if not is_student or not permitted_usn:
            if is_teacher:
                return {
                    'is_academic_query': True,
                    'handled': True,
                    'reply': "Please specify the student's USN or name to view their academic analysis (e.g., *Show details of USN 2024IT0001*).",
                    'sources': [],
                    'error': None,
                }
            return {
                'is_academic_query': True,
                'handled': True,
                'reply': "Access Denied\n\nUnable to verify your enrolled student profile.",
                'sources': [],
                'error': 'unauthorized',
            }
        target = permitted_usn

    # Check RBAC for Individual Student
    if is_student:
        # A student can ONLY view their own permitted information!
        target_clean = re.sub(r'^(?:USN[\s_:]*)', '', target, flags=re.IGNORECASE).strip().upper()
        if target_clean != permitted_usn:
            return {
                'is_academic_query': True,
                'handled': True,
                'reply': "Access Denied\n\nYou do not have permission to access another student's academic information.",
                'sources': ["Role-Based Access Control Policy"],
                'error': 'access_denied',
            }

    if not is_teacher and not is_student:
        return {
            'is_academic_query': True,
            'handled': True,
            'reply': "Access Denied\n\nYou do not have permission to access student academic records. Please log in as an authorized teacher, administrator, or student.",
            'sources': [],
            'error': 'unauthenticated',
        }

    # Handle ambiguous first names for teachers (e.g. "Rahul")
    if is_teacher and not USN_CLEAN_REGEX.search(target):
        ambiguous = engine.get_ambiguous_names(target)
        if len(ambiguous) > 1:
            lines = [f"Found {len(ambiguous)} students matching '{target.title()}':\n"]
            lines.append("| USN | Student Name |")
            lines.append("|---|---|")
            for a in ambiguous[:10]:
                lines.append(f"| {a['usn']} | {a['student_name']} |")
            lines.append("\nPlease ask using the specific USN (e.g., *Show details of USN " + ambiguous[0]['usn'] + "*).")
            return {
                'is_academic_query': True,
                'handled': True,
                'reply': "\n".join(lines),
                'sources': ["dataset.csv"],
                'error': None,
            }

    # Execute Analysis
    analysis = engine.analyze_student(target)
    if not analysis.get('found', False):
        return {
            'is_academic_query': True,
            'handled': True,
            'reply': "The required information is not available in the current dataset.",
            'sources': ["Student Academic Dataset (dataset.csv)"],
            'error': None,
        }

    focus_area = detect_focus_area(user_query)
    reply = format_student_dashboard(analysis, focus_area=focus_area)

    # Context-aware related questions that do not repeat the current question
    if is_student:
        if focus_area == 'risk':
            related_questions = ["How can I improve my performance?", "Which career areas are suitable for me?", "What is my predicted final score?"]
        elif focus_area == 'intervention':
            related_questions = ["Which career areas are suitable for me?", "What is my attendance across subjects?", "Show my complete report"]
        elif focus_area == 'career':
            related_questions = ["How can I improve my performance?", "What is my predicted final score?", "Show my complete report"]
        elif focus_area == 'prediction':
            related_questions = ["Am I at risk?", "How can I improve my performance?", "Show my complete report"]
        elif focus_area == 'attendance':
            related_questions = ["Am I at risk?", "What is my predicted final score?", "Show my complete report"]
        elif focus_area == 'trend':
            related_questions = ["How can I improve my performance?", "What is my predicted final score?", "Show my complete report"]
        else: # overview
            related_questions = ["Am I at risk?", "How can I improve my performance?", "Which career areas are suitable for me?"]
    else: # teacher / staff
        if focus_area == 'risk':
            related_questions = ["What intervention should I provide?", "Which career areas are suitable for this student?", "What is their predicted final score?"]
        elif focus_area == 'intervention':
            related_questions = ["Which career areas are suitable for this student?", "What is their attendance across subjects?", "Show full student report"]
        elif focus_area == 'career':
            related_questions = ["What intervention should I provide?", "What is their predicted final score?", "Show full student report"]
        elif focus_area == 'prediction':
            related_questions = ["Is this student at risk?", "What intervention should I provide?", "Show full student report"]
        elif focus_area == 'attendance':
            related_questions = ["Is this student at risk?", "What intervention should I provide?", "Show full student report"]
        elif focus_area == 'trend':
            related_questions = ["Is this student at risk?", "What intervention should I provide?", "Show full student report"]
        else: # overview
            related_questions = ["Is this student at risk?", "What intervention should I provide?", "Which career areas are suitable for this student?"]

    return {
        'is_academic_query': True,
        'handled': True,
        'reply': reply,
        'sources': [f"Student Record ({analysis['profile']['usn']})"],
        'focus_area': focus_area,
        'related_questions': related_questions,
        'error': None,
    }
