import os

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import (
    StudentProfile, TeacherProfile, VideoLecture,
    StudyMaterial, Quiz, QuizQuestion, Assignment,
    AssignmentSubmission, QuizAttempt, Subject
)


class SubjectForm(forms.ModelForm):
    """Form for teachers/admins to add or edit a subject."""
    _fc = {'class': 'form-control'}

    name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={**_fc, 'placeholder': 'e.g. Operating Systems'}),
        help_text='Full subject name'
    )
    code = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={**_fc, 'placeholder': 'e.g. OS', 'style': 'text-transform:uppercase'}),
        help_text='Short unique code (e.g. DS, PY, OS)'
    )
    description = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={**_fc, 'rows': 3, 'placeholder': 'Brief description (optional)'}),
    )

    class Meta:
        model = Subject
        fields = ['name', 'code', 'description']

    def clean_code(self):
        return self.cleaned_data.get('code', '').strip().upper()


def _validate_uploaded_file(
    uploaded_file,
    *,
    allowed_extensions,
    allowed_content_types,
    max_size_bytes,
    label,
):
    """Shared file validation for all upload forms."""
    if not uploaded_file:
        return uploaded_file

    ext = os.path.splitext(uploaded_file.name or "")[1].lower()
    if ext not in allowed_extensions:
        allowed = ', '.join(sorted(allowed_extensions))
        raise forms.ValidationError(
            f'Unsupported file type for {label}. Allowed: {allowed}.'
        )

    content_type = (getattr(uploaded_file, 'content_type', '') or '').lower()
    if content_type and content_type not in allowed_content_types:
        raise forms.ValidationError(
            f'Unsupported MIME type for {label}: {content_type}.'
        )

    if uploaded_file.size > max_size_bytes:
        raise forms.ValidationError(
            f'{label.capitalize()} file is too large. Max size is {max_size_bytes // (1024 * 1024)} MB.'
        )

    return uploaded_file


class StudentRegistrationForm(UserCreationForm):
    _fc = {'class': 'form-control'}
    first_name   = forms.CharField(max_length=50, required=True,  widget=forms.TextInput(attrs={**_fc, 'autocomplete': 'given-name'}))
    last_name    = forms.CharField(max_length=50, required=True,  widget=forms.TextInput(attrs={**_fc, 'autocomplete': 'family-name'}))
    email        = forms.EmailField(required=True,                 widget=forms.EmailInput(attrs={**_fc, 'autocomplete': 'email'}))
    roll_number  = forms.CharField(max_length=20,                  widget=forms.TextInput(attrs={**_fc, 'autocomplete': 'off'}))
    semester     = forms.IntegerField(min_value=1, max_value=8,    widget=forms.NumberInput(attrs={**_fc, 'autocomplete': 'off'}))
    phone        = forms.CharField(max_length=15, required=False,  widget=forms.TextInput(attrs={**_fc, 'autocomplete': 'tel'}))
    previous_gpa = forms.FloatField(min_value=0.0, max_value=10.0, widget=forms.NumberInput(attrs={**_fc, 'step': '0.01', 'autocomplete': 'off'}))

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control'}),
        }

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError('An account with this email address already exists.')
        return email.lower()

    def clean_roll_number(self):
        roll = self.cleaned_data.get('roll_number')
        from .models import StudentProfile
        if StudentProfile.objects.filter(roll_number=roll).exists():
            raise forms.ValidationError('This roll number is already registered.')
        return roll

    def save(self, commit=True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
            StudentProfile.objects.create(
                user=user,
                roll_number=self.cleaned_data['roll_number'],
                semester=self.cleaned_data['semester'],
                phone=self.cleaned_data.get('phone', ''),
                previous_gpa=self.cleaned_data.get('previous_gpa', 0.0),
            )
        return user


class VideoLectureForm(forms.ModelForm):
    class Meta:
        model = VideoLecture
        fields = ['subject', 'title', 'description', 'video_file', 'youtube_url', 'duration_minutes']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }

    def clean(self):
        cleaned_data = super().clean()
        video_file = cleaned_data.get('video_file')
        youtube_url = cleaned_data.get('youtube_url')
        if not video_file and not youtube_url:
            raise forms.ValidationError(
                'Please provide either a video file or a YouTube URL.'
            )
        return cleaned_data

    def clean_video_file(self):
        video_file = self.cleaned_data.get('video_file')
        return _validate_uploaded_file(
            video_file,
            allowed_extensions={'.mp4', '.webm', '.mov', '.mkv', '.avi'},
            allowed_content_types={
                'video/mp4',
                'video/webm',
                'video/quicktime',
                'video/x-matroska',
                'video/x-msvideo',
                'application/octet-stream',
            },
            max_size_bytes=500 * 1024 * 1024,
            label='video',
        )


class StudyMaterialForm(forms.ModelForm):
    class Meta:
        model = StudyMaterial
        fields = ['subject', 'title', 'description', 'file']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }

    def clean_file(self):
        material_file = self.cleaned_data.get('file')
        return _validate_uploaded_file(
            material_file,
            allowed_extensions={'.pdf', '.doc', '.docx', '.ppt', '.pptx', '.txt', '.md'},
            allowed_content_types={
                'application/pdf',
                'application/msword',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/vnd.ms-powerpoint',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                'text/plain',
                'text/markdown',
                'application/octet-stream',
            },
            max_size_bytes=25 * 1024 * 1024,
            label='study material',
        )


class QuizForm(forms.ModelForm):
    class Meta:
        model = Quiz
        fields = ['subject', 'title', 'description', 'total_marks', 'time_limit_minutes', 'due_date']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'due_date': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
        }


class QuizQuestionForm(forms.ModelForm):
    class Meta:
        model = QuizQuestion
        fields = ['question_text', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_answer', 'marks']
        widgets = {
            'question_text': forms.Textarea(attrs={'rows': 2}),
        }


class AssignmentForm(forms.ModelForm):
    class Meta:
        model = Assignment
        fields = ['subject', 'title', 'description', 'total_marks', 'due_date', 'attachment']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 4}),
            'due_date': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
        }

    def clean_attachment(self):
        attachment = self.cleaned_data.get('attachment')
        return _validate_uploaded_file(
            attachment,
            allowed_extensions={'.pdf', '.doc', '.docx', '.ppt', '.pptx', '.txt', '.zip', '.py'},
            allowed_content_types={
                'application/pdf',
                'application/msword',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/vnd.ms-powerpoint',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                'text/plain',
                'application/zip',
                'application/x-zip-compressed',
                'text/x-python',
                'application/octet-stream',
            },
            max_size_bytes=25 * 1024 * 1024,
            label='assignment attachment',
        )


class AssignmentSubmissionForm(forms.ModelForm):
    class Meta:
        model = AssignmentSubmission
        fields = ['submitted_file']

    def clean_submitted_file(self):
        submitted_file = self.cleaned_data.get('submitted_file')
        return _validate_uploaded_file(
            submitted_file,
            allowed_extensions={
                '.pdf', '.doc', '.docx', '.txt', '.zip', '.py', '.ipynb',
                '.java', '.c', '.cpp', '.js', '.html', '.css', '.sql',
            },
            allowed_content_types={
                'application/pdf',
                'application/msword',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'text/plain',
                'application/zip',
                'application/x-zip-compressed',
                'application/x-ipynb+json',
                'text/x-python',
                'text/x-java-source',
                'text/x-c',
                'text/javascript',
                'text/html',
                'text/css',
                'application/sql',
                'application/octet-stream',
            },
            max_size_bytes=25 * 1024 * 1024,
            label='submission',
        )


class GradeSubmissionForm(forms.ModelForm):
    class Meta:
        model = AssignmentSubmission
        fields = ['marks_obtained', 'feedback']
        widgets = {
            'feedback': forms.Textarea(attrs={'rows': 3}),
        }

    def __init__(self, *args, **kwargs):
        self._total_marks = kwargs.pop('total_marks', None)
        super().__init__(*args, **kwargs)

    def clean_marks_obtained(self):
        value = self.cleaned_data.get('marks_obtained')
        if value is None:
            return value
        if value < 0:
            raise forms.ValidationError('Marks cannot be negative.')
        if self._total_marks is not None and value > self._total_marks:
            raise forms.ValidationError(
                f'Marks cannot exceed total marks ({self._total_marks}).'
            )
        return value
