
import joblib
import pandas as pd
import sys
from pathlib import Path
import streamlit as st

# Ket noi cau hinh va du lieu chung cua nhom
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.data import load_data

st.set_page_config(
    page_title="Du doan ket qua hoc tap",
    page_icon="🎓",
    layout="wide"
)

@st.cache_data
def get_reference_data():
    return load_data()

df = get_reference_data()

st.title("🎓 DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN")
st.write("Hệ thống dự đoán nguy cơ bỏ học bằng Machine Learning.")

st.subheader("1. Chọn thời điểm dự đoán")

feature_set = st.selectbox(
    "Tập đặc trưng",
    ["A", "B", "C"],
    format_func=lambda x: {
        "A": "A - Khi nhập học (24 đặc trưng)",
        "B": "B - Sau học kỳ 1 (30 đặc trưng)",
        "C": "C - Sau học kỳ 2 (36 đặc trưng)"
    }[x]
)

features = config.FEATURE_SETS[feature_set]

st.success(
    f"Đã chọn tập {feature_set} - "
    f"{len(features)} đặc trưng"
)

st.subheader("2. Nhập thông tin sinh viên")

values = {}

with st.form("student_form"):
    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):
        current_col = col1 if i % 2 == 0 else col2

        with current_col:
            if feature in config.CATEGORICAL_COLS:
                options = sorted(
                    df[feature].dropna().unique().tolist()
                )
                values[feature] = st.selectbox(
                    feature,
                    options,
                    key=f"{feature_set}_{feature}"
                )

            elif feature in config.BINARY_COLS:
                values[feature] = st.selectbox(
                    feature,
                    [0, 1],
                    key=f"{feature_set}_{feature}"
                )

          
            else:
                default = float(df[feature].median())

                if feature == "Age at enrollment":
                    values[feature] = st.number_input(
                        feature,
                        min_value=1,
                        value=max(1, int(default)),
                        step=1,
                        key=f"{feature_set}_{feature}"
                    )

                elif feature.startswith("Curricular units") and "(grade)" not in feature:
                    values[feature] = st.number_input(
                        feature,
                        min_value=0,
                        value=max(0, int(default)),
                        step=1,
                        key=f"{feature_set}_{feature}"
                    )

                elif feature.startswith("Curricular units") and "(grade)" in feature:
                    values[feature] = st.number_input(
                        feature,
                        min_value=0.0,
                        max_value=20.0,
                        value=min(20.0, max(0.0, default)),
                        step=0.1,
                        key=f"{feature_set}_{feature}"
                    )

                else:
                    values[feature] = st.number_input(
                        feature,
                        value=default,
                        step=0.1,
                        key=f"{feature_set}_{feature}"
                    )

    submitted = st.form_submit_button(
        "Kiểm tra dữ liệu đầu vào",
        use_container_width=True
    )


if submitted:
    
    # Kiem tra tinh hop ly cua so hoc phan
    errors = []

    for sem in ["1st", "2nd"]:
        enrolled = f"Curricular units {sem} sem (enrolled)"
        approved = f"Curricular units {sem} sem (approved)"

        if enrolled in values and approved in values:
            if values[approved] > values[enrolled]:
                errors.append(
                    f"Hoc ky {sem}: So hoc phan dat "
                    "khong duoc lon hon so hoc phan dang ky."
                )

    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    st.subheader("3. Kết quả dự đoán")

    st.success(f"Đã nhận {len(values)} đặc trưng đầu vào.")

    if feature_set != "C":
        st.warning(
            "Hiện tại mô hình thử nghiệm chỉ hỗ trợ tập C "
            "(36 đặc trưng). Hãy chọn tập C để dự đoán."
        )
    else:
        try:
            @st.cache_resource
            def load_model():
                return joblib.load(
                    ROOT / "models" / "gb_best.joblib"
                )

            model = load_model()

            input_df = pd.DataFrame(
                [values],
                columns=config.FEATURE_SETS["C"]
            )

            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_

            st.success(f"Kết quả dự đoán: {prediction}")

            st.subheader("Xác suất từng lớp")

            result_df = pd.DataFrame({
                "Lớp": classes,
                "Xác suất": probabilities
            })

            result_df["Tỷ lệ (%)"] = (
                result_df["Xác suất"] * 100
            ).round(2)

            st.dataframe(
                result_df[["Lớp", "Tỷ lệ (%)"]],
                hide_index=True,
                use_container_width=True
            )

            st.bar_chart(
                result_df.set_index("Lớp")["Xác suất"]
            )

            st.caption(
                "Kết quả thử nghiệm từ mô hình Gradient Boosting "
                "với tập C; không phải kết luận chắc chắn."
            )

        except Exception as e:
            st.error(f"Không thể thực hiện dự đoán: {e}")
