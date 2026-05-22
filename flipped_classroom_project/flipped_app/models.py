from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator


class Subject(models.Model):
    SUBJECT_CHOICES = [
        ('DS',   'Data Structures'),
        ('PY',   'Python Programming'),
        ('WD',   'Web Development'),
        ('CN',   'Computer Networks'),
        ('DSC',  'Data Science'),
        ('AIML', 'Artificial Intelligence & Machine Learning'),
    ]
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=10, choices=SUBJECT_CHOICES, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class StudentProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='student_profile')
    roll_number = models.CharField(max_length=20, unique=True)
    department = models.CharField(max_length=100, default='Computer Science & Engineering')
    semester = models.IntegerField(default=1, validators=[MinValueValidator(1), MaxValueValidator(8)])
    profile_picture = models.ImageField(upload_to='profile_pics/', blank=True, null=True)
    phone = models.CharField(max_length=15, blank=True)
    enrolled_subjects = models.ManyToManyField(Subject, blank=True)
    previous_gpa = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(10.0)]
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.get_full_name()} ({self.roll_number})"


class TeacherProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='teacher_profile')
    employee_id = models.CharField(max_length=20, unique=True)
    department = models.CharField(max_length=100, default='Computer Science & Engineering')
    designation = models.CharField(max_length=100, default='Assistant Professor')
    subjects = models.ManyToManyField(Subject, blank=True)

    def __str__(self):
        return f"Prof. {self.user.get_full_name()}"


class VideoLecture(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name='videos')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    video_file = models.FileField(upload_to='videos/', blank=True, null=True)
    youtube_url = models.URLField(blank=True)
    duration_minutes = models.FloatField(default=0)
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    @property
    def youtube_embed_url(self):
        """Convert any YouTube URL format to a privacy-enhanced embed URL."""
        url = self.youtube_url
        if not url:
            return ''
        # Already an embed URL – normalise to nocookie domain
        if 'embed/' in url:
            video_id = url.split('embed/')[-1].split('?')[0]
            return f'https://www.youtube-nocookie.com/embed/{video_id}'
        # Short URL: youtu.be/<id>
        if 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[-1].split('?')[0]
            return f'https://www.youtube-nocookie.com/embed/{video_id}'
        # Standard watch URL: ?v=<id>
        if 'watch?v=' in url:
            video_id = url.split('watch?v=')[-1].split('&')[0]
            return f'https://www.youtube-nocookie.com/embed/{video_id}'
        # Shorts: youtube.com/shorts/<id>
        if '/shorts/' in url:
            video_id = url.split('/shorts/')[-1].split('?')[0]
            return f'https://www.youtube-nocookie.com/embed/{video_id}'
        return url

    def __str__(self):
        return f"{self.subject.name} - {self.title}"


class StudyMaterial(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name='materials')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    file = models.FileField(upload_to='materials/')
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.subject.name} - {self.title}"


class Quiz(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name='quizzes')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    total_marks = models.IntegerField(default=10)
    time_limit_minutes = models.IntegerField(default=20)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    due_date = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.subject.name} - {self.title}"


class QuizQuestion(models.Model):
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='questions')
    question_text = models.TextField()
    option_a = models.CharField(max_length=300)
    option_b = models.CharField(max_length=300)
    option_c = models.CharField(max_length=300)
    option_d = models.CharField(max_length=300)
    correct_answer = models.CharField(
        max_length=1,
        choices=[('A', 'A'), ('B', 'B'), ('C', 'C'), ('D', 'D')]
    )
    marks = models.IntegerField(default=1)

    def __str__(self):
        return f"Q: {self.question_text[:60]}"


class Assignment(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name='assignments')
    title = models.CharField(max_length=200)
    description = models.TextField()
    total_marks = models.IntegerField(default=20)
    due_date = models.DateTimeField()
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    attachment = models.FileField(upload_to='assignments/', blank=True, null=True)

    def __str__(self):
        return f"{self.subject.name} - {self.title}"


class AssignmentSubmission(models.Model):
    assignment = models.ForeignKey(Assignment, on_delete=models.CASCADE, related_name='submissions')
    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name='submissions')
    submitted_file = models.FileField(upload_to='submissions/')
    submitted_at = models.DateTimeField(auto_now_add=True)
    marks_obtained = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(0)]
    )
    feedback = models.TextField(blank=True)
    is_graded = models.BooleanField(default=False)

    class Meta:
        unique_together = ('assignment', 'student')

    def __str__(self):
        return f"{self.student.username} - {self.assignment.title}"


class QuizAttempt(models.Model):
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='attempts')
    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name='quiz_attempts')
    score = models.FloatField(default=0)
    attempted_at = models.DateTimeField(auto_now_add=True)
    time_taken_minutes = models.FloatField(default=0)

    class Meta:
        unique_together = ('quiz', 'student')

    def __str__(self):
        return f"{self.student.username} - {self.quiz.title} ({self.score})"


class VideoWatchHistory(models.Model):
    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name='watch_history')
    video = models.ForeignKey(VideoLecture, on_delete=models.CASCADE)
    watched_at = models.DateTimeField(auto_now_add=True)
    watch_duration_minutes = models.FloatField(default=0)
    completed = models.BooleanField(default=False)

    class Meta:
        unique_together = ('student', 'video')

    def __str__(self):
        return f"{self.student.username} watched {self.video.title}"


class Attendance(models.Model):
    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name='attendance_records')
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    date = models.DateField()
    present = models.BooleanField(default=True)

    class Meta:
        unique_together = ('student', 'subject', 'date')

    def __str__(self):
        status = 'Present' if self.present else 'Absent'
        return f"{self.student.username} - {self.subject.name} - {self.date} - {status}"


class StudentPerformance(models.Model):
    """
    Aggregated performance metrics per student per subject.
    Used as primary data source for ML model.
    """
    PERFORMANCE_LABEL = [
        ('High', 'High Performer'),
        ('Medium', 'Medium Performer'),
        ('Low', 'Low Performer'),
        ('At-Risk', 'At-Risk Student'),
    ]

    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name='performance_records')
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)

    # Engagement Features
    videos_watched = models.IntegerField(default=0)
    total_video_time_minutes = models.FloatField(default=0)
    materials_downloaded = models.IntegerField(default=0)

    # Assessment Features
    quiz_avg_score = models.FloatField(default=0)
    assignment_avg_marks = models.FloatField(default=0)
    attendance_percentage = models.FloatField(default=0)
    participation_score = models.FloatField(default=0, validators=[MinValueValidator(0), MaxValueValidator(10)])

    # Academic Background
    previous_gpa = models.FloatField(default=0)

    # Outcome
    final_exam_score = models.FloatField(default=0, validators=[MinValueValidator(0), MaxValueValidator(100)])
    performance_label = models.CharField(max_length=10, choices=PERFORMANCE_LABEL, blank=True)
    predicted_score = models.FloatField(null=True, blank=True)
    predicted_label = models.CharField(max_length=10, blank=True)
    is_at_risk = models.BooleanField(default=False)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('student', 'subject')

    def __str__(self):
        return f"{self.student.username} - {self.subject.name} [{self.performance_label}]"

    def save(self, *args, **kwargs):
        # Auto-assign performance label only when a final score has been recorded
        if self.final_exam_score > 0:
            if self.final_exam_score >= 75:
                self.performance_label = 'High'
                self.is_at_risk = False
            elif self.final_exam_score >= 50:
                self.performance_label = 'Medium'
                self.is_at_risk = False
            elif self.final_exam_score >= 35:
                self.performance_label = 'Low'
                self.is_at_risk = False
            else:
                self.performance_label = 'At-Risk'
                self.is_at_risk = True
        super().save(*args, **kwargs)


class Notification(models.Model):
    recipient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Notification for {self.recipient.username}: {self.message[:50]}"


class ChatMessage(models.Model):
    """Stores FlipLearn AI chatbot conversation history."""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]
    student = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='chat_messages'
    )
    subject = models.ForeignKey(
        Subject, on_delete=models.SET_NULL, null=True, blank=True,
        help_text="Subject context for this message (optional)"
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    sources = models.TextField(blank=True, default="",
                               help_text="Comma-separated source names retrieved by RAG")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"[{self.role}] {self.student.username}: {self.content[:60]}"
