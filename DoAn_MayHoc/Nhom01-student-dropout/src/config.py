"""Cấu hình dùng chung cho toàn nhóm. KHÔNG tự ý sửa; cần đổi thì mở Issue/PR để nhóm trưởng review."""
from pathlib import Path

# ----- Đường dẫn -----
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_FILE = DATA_RAW / "data.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "reports"
CV_RESULTS_FILE = RESULTS_DIR / "cv_results.csv"
TEST_RESULTS_FILE = RESULTS_DIR / "test_results.csv"

# ----- Tái lập -----
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

# ----- Biến mục tiêu -----
TARGET = "Target"
CLASSES = ["Dropout", "Enrolled", "Graduate"]
POSITIVE_CLASS = "Dropout"  # lớp quan tâm nhất (Recall Dropout)

# ----- Chuẩn hóa tên cột (xem data.clean_columns) -----
# Tên gốc của UCI có lỗi chính tả "Nacionality" và một số cột có khoảng trắng/tab thừa.
COLUMN_RENAMES = {"Nacionality": "Nationality"}

# ----- Nhóm cột -----
# Biến phân loại mã hóa bằng số nguyên, KHÔNG có thứ tự -> One-Hot
CATEGORICAL_COLS = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nationality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# Biến nhị phân 0/1: giữ nguyên, không cần One-Hot
BINARY_COLS = [
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# Biến số (liên tục / đếm). "Application order" (0-9) coi là biến thứ tự -> để ở nhóm số.
NUMERIC_COLS = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

CURRICULAR_SUFFIXES = [
    "(credited)",
    "(enrolled)",
    "(evaluations)",
    "(approved)",
    "(grade)",
    "(without evaluations)",
]
CURRICULAR_1ST = [f"Curricular units 1st sem {s}" for s in CURRICULAR_SUFFIXES]
CURRICULAR_2ND = [f"Curricular units 2nd sem {s}" for s in CURRICULAR_SUFFIXES]

BASE_COLS = CATEGORICAL_COLS + BINARY_COLS + NUMERIC_COLS

# ----- 3 tập đặc trưng theo thời điểm dự đoán (RQ3) -----
FEATURE_SETS = {
    "A": BASE_COLS,                                  # chỉ thông tin lúc nhập học
    "B": BASE_COLS + CURRICULAR_1ST,                 # A + kết quả học kỳ 1
    "C": BASE_COLS + CURRICULAR_1ST + CURRICULAR_2ND,  # toàn bộ đặc trưng (36, xác minh với dữ liệu thật)
}
