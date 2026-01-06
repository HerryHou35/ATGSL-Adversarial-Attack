import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# ======================
# 1. 加载模型与数据
# ======================

with open(r"D:\pylearning\vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(r"D:\pylearning\svm_clf.pkl", "rb") as f:
    clf = pickle.load(f)

test_path = r"D:\Natural Language Processing\实验1\通话数据互动策略结果\测试集结果.csv"
df = pd.read_csv(test_path).dropna(subset=["is_fraud"])
df["is_fraud"] = df["is_fraud"].astype(int)

X = df["specific_dialogue_content"].astype(str)
y = df["is_fraud"]

feature_names = np.array(vectorizer.get_feature_names_out())

# ======================
# 2. 手工替换词表（可慢慢扩）
# ======================

replace_dict = {
    "为了您的资金安全": ["出于安全考虑", "为了保障安全"],
    "非常简单": ["不复杂", "很容易"],
    "李经理": ["客服人员", "工作人员"],
    "绝对安全": ["风险很低", "基本安全"],
    "填写相关信息": ["提交必要资料", "填写基本资料"],
    "需要您配合我们进行核实": ["需要进行简单核实"]
}

# ======================
# 3. 敏感词计算
# ======================

def get_topk_sensitive_words(text, k=5):
    x_vec = vectorizer.transform([text])
    original_score = clf.decision_function(x_vec)[0]
    x_dense = x_vec.toarray()[0]
    idxs = np.where(x_dense > 0)[0]

    sensitivities = []
    for i in idxs:
        x_mod = x_dense.copy()
        x_mod[i] = 0.0
        new_score = clf.decision_function([x_mod])[0]
        delta = abs(original_score - new_score)
        sensitivities.append((feature_names[i], delta))

    sensitivities.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in sensitivities[:k]]

# ======================
# 4. 攻击函数（ATGSL 核心）
# ======================
def attack(text, max_steps=5):
    current = text

    for _ in range(max_steps):
        words = get_topk_sensitive_words(current)

        best_candidate = current
        best_score = clf.decision_function(vectorizer.transform([current]))[0]

        for w in words:
            # 1️⃣ 删除
            if w in current:
                cand = current.replace(w, "")
                score = clf.decision_function(vectorizer.transform([cand]))[0]
                if abs(score) < abs(best_score):
                    best_score = score
                    best_candidate = cand

            # 2️⃣ 替换
            if w in replace_dict:
                for rep in replace_dict[w]:
                    if w in current:
                        cand = current.replace(w, rep)
                        score = clf.decision_function(vectorizer.transform([cand]))[0]
                        if abs(score) < abs(best_score):
                            best_score = score
                            best_candidate = cand

        if best_candidate == current:
            break

        current = best_candidate
        if clf.predict(vectorizer.transform([current]))[0] == 0:
            return True, current

    return False, current

# def attack(text, max_steps=5):
#     current = text
#
#     for _ in range(max_steps):
#         words = get_topk_sensitive_words(current)
#
#         best_candidate = current
#         best_score = clf.decision_function(vectorizer.transform([current]))[0]
#
#         for w in words:
#             if w not in replace_dict:
#                 continue
#             for rep in replace_dict[w]:
#                 if w not in current:
#                     continue
#                 cand = current.replace(w, rep)
#                 score = clf.decision_function(vectorizer.transform([cand]))[0]
#
#                 if abs(score) < abs(best_score):
#                     best_score = score
#                     best_candidate = cand
#
#         # 如果没有更优候选，停止
#         if best_candidate == current:
#             break
#
#         current = best_candidate
#
#         # 如果已经翻转成非诈骗
#         pred = clf.predict(vectorizer.transform([current]))[0]
#         if pred == 0:
#             return True, current
#
#     return False, current

# ======================
# 5. 执行攻击 & 统计
# ======================

success = 0
total = 0

results = []

print("开始 ATGSL 攻击测试...")

for i in tqdm(range(len(X))):
    if y.iloc[i] != 1:
        continue

    total += 1
    ok, adv = attack(X.iloc[i])

    if ok:
        success += 1

    results.append({
        "original": X.iloc[i],
        "adversarial": adv,
        "success": ok
    })

print("\n攻击成功率 ASR =", success / total)

pd.DataFrame(results).to_csv(r"D:\Natural Language Processing\实验1\atgsl_attack_results.csv",
                             index=False, encoding="utf-8-sig")
print("攻击样本已保存。")
