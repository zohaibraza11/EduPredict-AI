"""EduPredict AI - core AI module.
AI components (both written from scratch with NumPy):
  1. K-Nearest Neighbours (KNN) classifier
  2. Gaussian Naive Bayes classifier
Integration: soft voting (average of both probability outputs) = Ensemble.
"""
import os, pickle, time
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE, "training_data.csv")
STUDENTS_PATH = os.path.join(BASE, "student_records.csv")
MODEL_PATH = os.path.join(BASE, "model_v3.pkl")
FEATURES = ["attendance", "study_hours", "assignment_score", "quiz_score",
            "midterm_score", "sleep_hours", "previous_gpa"]
LABELS = ["At Risk", "Average", "Good", "Excellent"]
NAMES = ["Ali Raza","Ayesha Khan","Hamza Malik","Fatima Noor","Usman Tariq","Zainab Ahmed","Bilal Hussain","Maryam Iqbal","Hassan Ali","Sana Javed","Ahmed Raza","Hira Aslam","Talha Mahmood","Noor Fatima","Saad Ahmad","Iqra Shahid","Danish Butt","Laiba Khalid","Faizan Qureshi","Amna Siddiqui","Umer Farooq","Rabia Anwar","Shahid Mehmood","Sidra Yousaf","Kamran Akbar","Mahnoor Saleem","Waqas Ahmad","Areeba Zahid","Junaid Sheikh","Komal Rashid","Imran Haider","Anam Gul","Rizwan Ali","Hina Pervaiz","Arslan Mirza","Nimra Ashraf","Adeel Chaudhry","Sobia Kanwal","Naveed Aslam","Mehwish Ali","Fahad Nawaz","Tayyaba Riaz","Owais Raza","Bushra Tahir","Haris Javed","Rida Imran","Moiz Ahmed","Saima Bashir","Zeeshan Khan","Alishba Hassan"]


# ---------------------------------------------------------------- data
def generate_data(n=3000, seed=42):
    """Generates sample student data (NumPy + Pandas) and labels it from a weighted score."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "attendance": rng.normal(80, 12, n).clip(30, 100),
        "study_hours": rng.normal(3, 1.5, n).clip(0, 10),
        "assignment_score": rng.normal(70, 15, n).clip(0, 100),
        "quiz_score": rng.normal(68, 16, n).clip(0, 100),
        "midterm_score": rng.normal(65, 17, n).clip(0, 100),
        "sleep_hours": rng.normal(7, 1.2, n).clip(3, 10),
        "previous_gpa": rng.normal(2.8, 0.6, n).clip(1, 4),
    }).round(2)
    score = (0.20 * df.attendance + 0.10 * df.study_hours * 10 + 0.15 * df.assignment_score
             + 0.15 * df.quiz_score + 0.20 * df.midterm_score + 0.05 * df.sleep_hours * 10
             + 0.15 * df.previous_gpa * 25 + rng.normal(0, 1, n))  # noise = real-life unpredictability
    df["performance"] = pd.cut(score, [-np.inf, 62, 70, 77, np.inf], labels=LABELS).astype(str)
    return df


def ensure_students():
    """Creates the 50 pre-saved student records on first run."""
    if os.path.exists(STUDENTS_PATH):
        return
    df = generate_data(len(NAMES), seed=7)
    df.insert(0, "name", NAMES)
    df.insert(0, "id", [f"S{i:03d}" for i in range(1, len(NAMES) + 1)])
    df["added_on"] = "Pre-saved"
    df.to_csv(STUDENTS_PATH, index=False)


def get_students():
    ensure_students()
    return pd.read_csv(STUDENTS_PATH)


def add_student(name, vals, label):
    """Saves a newly predicted student permanently."""
    df = get_students()
    row = {"id": f"S{len(df) + 1:03d}", "name": name, **vals, "performance": label,
           "added_on": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    pd.concat([df, pd.DataFrame([row])]).to_csv(STUDENTS_PATH, index=False)
    return row["id"]


# ------------------------------------------------- AI component 1: KNN
class KNN:
    """Predicts by majority vote of the k closest students (Euclidean distance)."""
    def __init__(self, k=7):
        self.k = k

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9      # standardise features
        self.X, self.y = (X - self.mu) / self.sd, np.asarray(y)
        return self

    def proba(self, X):
        out = []
        for i in range(0, len(X), 500):                    # chunks keep memory low
            Z = (X[i:i + 500] - self.mu) / self.sd
            d = (Z ** 2).sum(1)[:, None] - 2 * Z @ self.X.T + (self.X ** 2).sum(1)[None, :]
            votes = self.y[np.argsort(d, axis=1)[:, :self.k]]
            out += [[(v == c).mean() for c in LABELS] for v in votes]
        return np.array(out)


# ---------------------------------------- AI component 2: Naive Bayes
class NaiveBayes:
    """Gaussian Naive Bayes: P(class) x product of P(feature | class)."""
    def fit(self, X, y):
        y = np.asarray(y)
        self.prior = np.array([(y == c).mean() for c in LABELS])
        self.mean = np.array([X[y == c].mean(0) for c in LABELS])
        self.var = np.array([X[y == c].var(0) + 1e-6 for c in LABELS])
        return self

    def proba(self, X):
        ll = -0.5 * (np.log(2 * np.pi * self.var)[None] + (X[:, None, :] - self.mean[None]) ** 2 / self.var[None]).sum(2)
        ll += np.log(self.prior + 1e-12)[None]
        ll -= ll.max(1, keepdims=True)                     # numerical stability
        p = np.exp(ll)
        return p / p.sum(1, keepdims=True)


# ------------------------------------------------------- evaluation
def metrics(y, pred):
    """Accuracy, per-class and macro precision/recall/F1, confusion matrix."""
    cm = pd.DataFrame(0, index=LABELS, columns=LABELS)
    for a, p in zip(y, pred):
        cm.loc[a, p] += 1
    tp = np.diag(cm.values).astype(float)
    prec = tp / np.maximum(cm.sum(0).values, 1)
    rec = tp / np.maximum(cm.sum(1).values, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    cm.index, cm.columns = [f"Actual {l}" for l in LABELS], [f"Pred {l}" for l in LABELS]
    return {"accuracy": tp.sum() / len(y), "precision": prec.mean(), "recall": rec.mean(),
            "f1": f1.mean(), "cm": cm,
            "per_class": pd.DataFrame({"precision": prec, "recall": rec, "f1": f1}, index=LABELS).round(3)}


def train():
    """Trains both models on 80% of the data, tests on the unseen 20%, saves everything."""
    df = generate_data()
    df.to_csv(TRAIN_PATH, index=False)
    ensure_students()
    X, y = df[FEATURES].values, df["performance"].values
    idx = np.random.default_rng(42).permutation(len(df))
    cut = int(0.8 * len(df))
    tr, te = idx[:cut], idx[cut:]
    models = {"KNN": KNN(7).fit(X[tr], y[tr]), "Naive Bayes": NaiveBayes().fit(X[tr], y[tr])}
    probs, ev = {}, {}
    for name, mod in models.items():
        t = time.perf_counter()
        probs[name] = mod.proba(X[te])
        dt = time.perf_counter() - t
        ev[name] = metrics(y[te], np.array(LABELS)[probs[name].argmax(1)])
        ev[name]["time"] = dt
    ens = (probs["KNN"] + probs["Naive Bayes"]) / 2
    ev["Ensemble"] = metrics(y[te], np.array(LABELS)[ens.argmax(1)])
    ev["Ensemble"]["time"] = ev["KNN"]["time"] + ev["Naive Bayes"]["time"]
    bundle = {"models": models, "eval": ev, "n_train": len(tr), "n_test": len(te)}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return ev["Ensemble"]["accuracy"]


def load_model():
    """Loads the saved models; retrains automatically if missing or incompatible."""
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        train()
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)


# ------------------------------------------------------- prediction
def _clean(df):
    """Validates input: all features present, numeric, no missing values."""
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Feature values must be numeric and not empty.")
    return X.values.astype(float)


def predict(bundle, row):
    """Ensemble prediction for one student: (label, probability per class)."""
    X = _clean(row)
    p = sum(mod.proba(X) for mod in bundle["models"].values())[0] / 2
    return LABELS[int(p.argmax())], dict(zip(LABELS, p))


def predict_models(bundle, row):
    """Each individual model's prediction for one student."""
    X = _clean(row)
    return {n: LABELS[int(mod.proba(X)[0].argmax())] for n, mod in bundle["models"].items()}


def predict_df(bundle, df):
    """Ensemble prediction for many students (batch)."""
    X = _clean(df)
    p = sum(mod.proba(X) for mod in bundle["models"].values()) / 2
    return np.array(LABELS)[p.argmax(1)]


def suggestions(r: dict):
    """Simple advice based on weak areas."""
    tips = []
    if r["attendance"] < 75: tips.append("Keep attendance above 75% and avoid missing classes.")
    if r["study_hours"] < 2.5: tips.append("Build a routine of at least 3 hours of study per day.")
    if r["assignment_score"] < 60: tips.append("Complete assignments on time and with care.")
    if r["quiz_score"] < 60: tips.append("Revise regularly for quizzes.")
    if r["midterm_score"] < 55: tips.append("Review weak midterm topics again with your teacher.")
    if r["sleep_hours"] < 6: tips.append("Sleep 7-8 hours to improve focus.")
    return tips or ["Great job! Keep up the consistency."]


if __name__ == "__main__":
    print(f"Models trained. Ensemble test accuracy: {train():.2%}")
