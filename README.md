# EduPredict AI - Student Performance Detection System
AI components (written from scratch with NumPy): KNN + Gaussian Naive Bayes, combined by soft voting.
```
pip install -r requirements.txt
python -m streamlit run app.py
```
- `model.py`: data, KNN, Naive Bayes, ensemble, evaluation, record saving
- `app.py`: user interface
- `student_records.csv`: 50 pre-saved students (new predictions are added automatically)
- `training_data.csv`: 3000 sample students used for training/testing
