import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# ============================
# 1. 数据路径
# ============================
train_path = r"D:\Natural Language Processing\实验1\通话数据互动策略结果\训练集结果.csv"
test_path  = r"D:\Natural Language Processing\实验1\通话数据互动策略结果\测试集结果.csv"

# ============================
# 2. 读取数据
# ============================
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

train_df = train_df.dropna(subset=["is_fraud"])
test_df  = test_df.dropna(subset=["is_fraud"])

train_df["is_fraud"] = train_df["is_fraud"].astype(int)
test_df["is_fraud"]  = test_df["is_fraud"].astype(int)

X_train = train_df["specific_dialogue_content"].astype(str)
y_train = train_df["is_fraud"]
X_test  = test_df["specific_dialogue_content"].astype(str)
y_test  = test_df["is_fraud"]

print("训练集样本数:", len(X_train))
print("测试集样本数:", len(X_test))

# ============================
# 3. TF-IDF 向量化
# ============================
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print("向量维度:", X_train_vec.shape)

# ============================
# 4. 训练 SVM
# ============================
clf = LinearSVC()
clf.fit(X_train_vec, y_train)

acc = clf.score(X_test_vec, y_test)
print("SVM 训练完成，准确率:", acc)

# ============================
# 5. 保存模型（🔥关键）
# ============================
save_dir = r"D:\pylearning"

with open(save_dir + r"\vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open(save_dir + r"\svm_clf.pkl", "wb") as f:
    pickle.dump(clf, f)

print("模型已保存到:", save_dir)
print("  - vectorizer.pkl")
print("  - svm_clf.pkl")
