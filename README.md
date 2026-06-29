# 📘 Flipped Classroom Using RAG: Performance Analysis in CSE Education

## 📌 Project Overview
This project is a web-based learning platform designed to implement and analyze the **Flipped Classroom** model in **Computer Science and Engineering (CSE)** education. Students access study materials, watch video lectures, and attempt quizzes/assignments before class. The system uses Machine Learning models to analyze interaction data in real-time, predict final exam scores, and identify at-risk students. Additionally, it integrates a Retrieval-Augmented Generation (RAG) agent (**FlipLearn AI**) to answer student queries grounded in course textbooks and slides.

---

## 🎯 Objectives
- **Implement a Flipped Classroom Platform**: Allow students to access pre-class study materials, submit assignments, and take quizzes.
- **Collect Interaction Metrics**: Log student engagement metrics such as video watch duration, downloads, and chatbot queries.
- **Predict Student Performance**: Run `scikit-learn` Machine Learning models to predict final exam scores and classify performance levels.
- **Identify At-Risk Students Early**: Automatically flag students performing below expected thresholds and send targeted alerts.
- **Grounded Chatbot Tutor**: Provide a RAG-based AI Tutor for academic content questions using the platform's knowledge base.

---

## 🏗 System Architecture & Key Files

### 1. Django Web Application
The core web interface and backend are located in the [flipped_app](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app) directory.
* **Database Models**: Defined in [models.py](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py).
  * [StudentProfile](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L16) & [TeacherProfile](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L34): Track user metadata.
  * [VideoLecture](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L45) & [VideoWatchHistory](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L175): Record lecture viewings.
  * [StudyMaterial](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L84): Tracks materials downloaded.
  * [QuizAttempt](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L161) & [AssignmentSubmission](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L142): Track academic marks.
  * [Attendance](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L189): Tracks presence in class.
  * [StudentPerformance](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L203): Aggregates engagement and performance data.
* **Routing & Views**: URLs are configured in [urls.py](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/urls.py), and view logic is in [views.py](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/views.py).
* **Automatic Engagement Updates**: Whenever students interact with quizzes, watch videos, or submit assignments, the [_update_engagement](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/views.py#L1058) helper updates aggregates and automatically runs the ML model to update prediction labels.

### 2. RAG AI Tutor (FlipLearn AI)
The academic chatbot engine is located in the [rag_engine](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/rag_engine) directory.
* **Embeddings**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) in [embedding_model.py](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/rag_engine/embedding_model.py) to represent text segments.
* **Retrieval**: Performs fast similarity searching against a FAISS index in [retriever.py](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/rag_engine/retriever.py).
* **Response Generation**: Queries the **Groq API** (`llama-3.1-8b-instant`) in [chat.py](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/rag_engine/chat.py).
* **Out-of-Domain (OOD) Protection**: Checks retrieve chunk scores; if the max similarity score is $< 0.38$, the chatbot answers strictly with: *"Your matched query is not found in our database."*
* **File Chat (PDF/Word)**: Supports uploading and extracting text from PDF and Word (`.docx`) files using the [chat_pdf_view](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/views.py#L1345) to stream explanations.

### 3. Machine Learning Module
The performance prediction models are located in the [ml_model](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/ml_model) directory.
* **Models**: Features are scaled via a `StandardScaler`, and predictions are made using `RandomForestRegressor` and `RandomForestClassifier`.
  * **Regressor**: Predicts final exam scores (scaled between 0-100).
  * **Classifier**: Classifies students into performance labels: *High*, *Medium*, *Low*, and *At-Risk*.
  * The logic resides in [prediction.py](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/ml_model/prediction.py).
* **Visualizations**: The script [regenerate_plots.py](file:///c:/Users/uttam/Downloads/RAG/regenerate_plots.py) generates confusion matrices, feature importances, and distribution graphs for the teacher analytics dashboard.

---

## 📊 Dataset Features & ML Targets
We collect and scale seven primary engagement and academic background features:
1. `videos_watched`: Total number of video lectures marked completed.
2. `total_video_time_minutes`: Total duration spent watching lectures.
3. `quiz_avg_score`: Average score across all attempted quizzes.
4. `assignment_avg_marks`: Average marks obtained on graded assignments.
5. `attendance_percentage`: Percentage of present records in subject classes.
6. `participation_score`: Participation points derived from AI Chatbot usage (0.5 points per query, capped at 10.0).
7. `previous_gpa`: The student's academic GPA from the previous semester.

---

## 🛠 Setup & Installation

### 1) Clone the Repository & Configure Environment
```bash
git clone https://github.com/your-username/flipped-classroom-ml.git
cd flipped-classroom-ml/flipped_classroom_project
```
Create a `.env` file in the `flipped_classroom_project` directory (adjacent to `manage.py`) with your Groq credentials:
```env
DEBUG=True
SECRET_KEY=dev-only-secret-key-change-before-deploy
GROQ_API_KEY=gsk_your_groq_api_key_here
RAG_ENABLE_WEB_SEARCH=False
```

### 2) Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3) Set Up the Database & Run Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 4) Train ML Models & Run Platform
To pre-train the random forest classifiers and regressors, run the training scripts:
```bash
python ml_model/model_training.py
```
You can also regenerate plots using:
```bash
python ../regenerate_plots.py
```
Finally, start the Django development server:
```bash
python manage.py runserver
```
Visit `http://127.0.0.1:8000/` in your browser.

---

## 📈 Evaluation Metrics
The ML pipeline logs training and validation statistics on every run, outputting:
* **Accuracy, Precision, Recall, and F1-Score** for classification.
* **Mean Squared Error (MSE)** and **$R^2$ Score** for regression.
* **Feature Importance Chart** demonstrating which metrics (e.g., quiz score, attendance, or previous GPA) affect student outcomes the most.

---

## 👨‍💻 Author
**Uttam Vitthal Bhise**  
M.Tech – Computer Science & Engineering
