from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np

def downstream_logistic(real_df, synth_df, label_col):
    # 1️⃣ 拆分特征 / 标签
    X_real = real_df.drop(columns=[label_col]).values
    y_real = real_df[label_col].values

    X_synth = synth_df.drop(columns=[label_col]).values
    y_synth = synth_df[label_col].values

    # 2️⃣ 标准化（用 synth 的统计量，符合训练流程）
    scaler = StandardScaler()
    X_synth = scaler.fit_transform(X_synth)
    X_real = scaler.transform(X_real)

    # 3️⃣ Logistic Regression（class-balanced）
    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_synth, y_synth)

    y_pred = clf.predict(X_real)

    return {
        "downstream_acc": accuracy_score(y_real, y_pred),
        "downstream_f1": f1_score(y_real, y_pred, average="binary"),
    }
