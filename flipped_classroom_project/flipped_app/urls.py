from django.urls import path
from . import views

urlpatterns = [
    # Auth
    path('', views.home_view, name='home'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Dashboard
    path('dashboard/', views.dashboard_view, name='dashboard'),

    # Subjects
    path('subjects/', views.subject_list_view, name='subjects'),
    path('subjects/enroll/<int:subject_id>/', views.enroll_subject_view, name='enroll'),

    # Video Lectures
    path('videos/', views.video_list_view, name='videos'),
    path('videos/upload/', views.upload_video_view, name='upload_video'),
    path('videos/subject/<int:subject_id>/', views.video_list_view, name='videos_by_subject'),
    path('videos/<int:video_id>/', views.video_detail_view, name='video_detail'),
    path('videos/<int:video_id>/mark-watched/', views.mark_video_watched_view, name='mark_video_watched'),

    # Study Materials
    path('materials/', views.material_list_view, name='materials'),
    path('materials/subject/<int:subject_id>/', views.material_list_view, name='materials_by_subject'),
    path('materials/upload/', views.upload_material_view, name='upload_material'),
    path('materials/download/<int:material_id>/', views.download_material_view, name='download_material'),

    # Quizzes
    path('quizzes/', views.quiz_list_view, name='quizzes'),
    path('quizzes/take/<int:quiz_id>/', views.take_quiz_view, name='take_quiz'),
    path('quizzes/result/<int:quiz_id>/', views.quiz_result_view, name='quiz_result'),
    path('quizzes/create/', views.create_quiz_view, name='create_quiz'),
    path('quizzes/<int:quiz_id>/add-question/', views.add_question_view, name='add_question'),

    # Assignments
    path('assignments/', views.assignment_list_view, name='assignments'),
    path('assignments/create/', views.create_assignment_view, name='create_assignment'),
    path('assignments/<int:assignment_id>/submissions/', views.assignment_submissions_view, name='assignment_submissions'),
    path('assignments/submit/<int:assignment_id>/', views.submit_assignment_view, name='submit_assignment'),
    path('assignments/grade/<int:submission_id>/', views.grade_submission_view, name='grade_submission'),

    # Analytics
    path('analytics/', views.analytics_view, name='analytics'),
    path('analytics/student/<int:student_id>/', views.student_detail_analytics_view, name='student_analytics'),
    path('analytics/run-ml/', views.run_ml_prediction_view, name='run_ml'),
    path('analytics/export-csv/', views.export_performance_csv, name='export_csv'),

    # Student performance
    path('my-performance/', views.my_performance_view, name='my_performance'),

    # Notifications
    path('notifications/read/<int:notif_id>/', views.mark_notification_read, name='mark_read'),

    # RAG
    path('rag/rebuild/', views.rebuild_rag_view, name='rebuild_rag'),

    # RAG Chatbot
    path('chat/ask/', views.chat_ask_view, name='chat_ask'),
    path('chat/history/', views.chat_history_view, name='chat_history'),
    path('chat/stream/', views.chat_stream_view, name='chat_stream'),
    path('chat/pdf/', views.chat_pdf_view, name='chat_pdf'),

    # Feedback
    path('feedback/', views.feedback_view, name='feedback'),

    # Video Comments
    path('videos/<int:video_id>/comment/', views.add_video_comment_view, name='add_video_comment'),
    path('videos/comment/<int:comment_id>/delete/', views.delete_video_comment_view, name='delete_video_comment'),

    # Student Accounts (Teacher / Admin)
    path('students/', views.student_accounts_view, name='student_accounts'),

    # Attendance
    path('attendance/mark/', views.mark_attendance_view, name='mark_attendance'),
    path('attendance/history/', views.attendance_history_view, name='attendance_history'),
    path('attendance/my/', views.my_attendance_view, name='my_attendance'),
]
