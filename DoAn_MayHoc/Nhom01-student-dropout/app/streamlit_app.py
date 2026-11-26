"""Interactive demonstration of the five student-outcome classifiers."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import config
from src.data import get_train_test
from src.predict import InputError, MODEL_FILES, load_metadata, predict_one
from src.preprocessing import get_feature_names


MODEL_LABELS = {
    "lr": "Logistic Regression",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "svm": "Support Vector Machine (RBF)",
}


@st.cache_data(show_spinner=False)
def training_defaults(feature_set: str) -> dict:
    X_train, _, _, _ = get_train_test(feature_set)
    return X_train.iloc[0].to_dict()


def show_feature_summary(model_key: str, feature_set: str, predicted_class: str):
    from src.predict import load_model

    pipeline = load_model(model_key, feature_set)
    clf = pipeline.named_steps["clf"]
    names = get_feature_names(pipeline)
    if hasattr(clf, "feature_importances_"):
        weights = pd.Series(clf.feature_importances_, index=names)
        title = "Yếu tố quan trọng toàn cục của mô hình"
        note = "Importance dựa trên impurity của cây; chỉ mô tả mô hình, không phải nguyên nhân hay giải thích cá thể."
    elif hasattr(clf, "coef_"):
        class_index = list(clf.classes_).index(predicted_class)
        weights = pd.Series(clf.coef_[class_index], index=names)
        title = f"Hệ số Logistic Regression cho lớp {predicted_class}"
        note = "Hệ số dương làm tăng điểm số tương đối của lớp này; không diễn giải thành quan hệ nhân quả."
    else:
        st.info("SVM RBF dùng kernel phi tuyến; không có hệ số/importance trực tiếp để trình bày.")
        return
    top = weights.reindex(weights.abs().sort_values(ascending=False).head(12).index)
    st.subheader(title)
    st.caption(note)
    st.dataframe(top.rename("Giá trị").to_frame(), width="stretch")


def main():
    st.set_page_config(page_title="Dự đoán kết quả học tập", page_icon="🎓", layout="wide")
    st.title("Dự đoán kết quả học tập của sinh viên")
    st.caption("Demo học thuật trên dữ liệu UCI. Kết quả không dùng để ra quyết định cá nhân.")
    st.info("Chọn thời điểm dự đoán A/B/C và mô hình. Mặc định dùng quy tắc native_predict; mọi xác suất được hiển thị theo thứ tự lớp cố định.")

    metadata = load_metadata()
    controls = st.columns(2)
    feature_set = controls[0].selectbox(
        "Tập đặc trưng / thời điểm dự đoán", list(config.FEATURE_SETS), index=2,
        format_func=lambda x: {
            "A": "A — thông tin lúc nhập học",
            "B": "B — thêm kết quả học kỳ 1",
            "C": "C — thêm kết quả học kỳ 2",
        }[x], key="feature_set",
    )
    model_key = controls[1].selectbox(
        "Mô hình", list(MODEL_FILES), index=3,
        format_func=lambda x: MODEL_LABELS[x], key="model",
    )
    st.write(f"Đang nhập **{len(config.FEATURE_SETS[feature_set])}** biến của tập {feature_set}.")

    defaults = training_defaults(feature_set)
    record = {}
    with st.form("student_prediction"):
        columns = st.columns(2)
        for index, name in enumerate(config.FEATURE_SETS[feature_set]):
            slot = columns[index % 2]
            default = defaults[name]
            if name in config.CATEGORICAL_COLS:
                options = metadata["known_categories"][name]
                default = int(default)
                choice_index = options.index(default) if default in options else 0
                record[name] = slot.selectbox(name, options, index=choice_index, key=f"input_{name}")
            elif name in config.BINARY_COLS:
                record[name] = slot.selectbox(name, [0, 1], index=int(default), key=f"input_{name}")
            else:
                step = 0.01 if "grade" in name.lower() or "rate" in name.lower() or "gdp" in name.lower() else 1.0
                record[name] = slot.number_input(name, value=float(default), step=step, key=f"input_{name}")
        submitted = st.form_submit_button("Dự đoán")

    if submitted:
        try:
            result = predict_one(record, model=model_key, feature_set=feature_set)
            left, right = st.columns([1, 2])
            left.metric("Nhãn dự đoán", result["label"])
            left.caption(f"Quy tắc: {result['decision_rule']} · Mô hình: {MODEL_LABELS[model_key]} · Tập: {feature_set}")
            probs = pd.DataFrame({"Lớp": config.CLASSES,
                                  "Xác suất": [result["probabilities"][c] for c in config.CLASSES]})
            right.dataframe(probs, hide_index=True, width="stretch")
            right.bar_chart(probs.set_index("Lớp"), y="Xác suất")
            for message in result["warnings"]:
                st.warning(message)
            show_feature_summary(model_key, feature_set, result["label"])
        except (InputError, FileNotFoundError, KeyError, ValueError) as exc:
            st.error(str(exc))

    with st.expander("Giới hạn và nguồn"):
        st.write("Các mô hình dùng cùng train/CV và cấu hình đã chọn trên tập C. Hold-out chưa được dùng trong demo. Importance toàn cục không giải thích riêng từng cá nhân.")
        st.write("Nguồn dữ liệu: UCI Machine Learning Repository, dataset 697.")


if __name__ == "__main__":
    main()
