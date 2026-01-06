import joblib
import numpy as np
import pandas as pd
import jieba
import os
from gensim.models import KeyedVectors
from tqdm import tqdm

# ==================== 1. 路径配置 ====================
MODEL_DIR = r"D:\pylearning"
FASTTEXT_PATH = r"D:\nlp_models\fasttext\cc.zh.300.vec"
DATA_PATH = r"D:\Natural Language Processing\实验1\通话数据互动策略结果\训练集结果.csv"
SAVE_DIR = r"D:\Natural Language Processing\大作业"
TEXT_COL = "specific_dialogue_content"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ==================== 2. 环境初始化 ====================
print(">>> [1/4] 正在加载 SVM 模型与词向量 (请稍候)...")
clf = joblib.load(os.path.join(MODEL_DIR, "svm_clf.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))

# 加载词向量
wv = KeyedVectors.load_word2vec_format(FASTTEXT_PATH, binary=False, limit=200000)

feature_names = vectorizer.get_feature_names_out()
weights = clf.coef_[0]
word_weight_map = dict(zip(feature_names, weights))


# ==================== 3. 辅助函数定义 (修复 NameError) ====================
def get_score(text):
    """计算 SVM 的决策分数"""
    processed = " ".join(jieba.lcut(str(text)))
    vec = vectorizer.transform([processed])
    return clf.decision_function(vec)[0]


# ==================== 4. 自动生成替换词表 ====================
print(">>> [2/4] 正在根据模型权重自动提取语义候选词...")
top_indices = np.argsort(weights)[::-1][:500]
top_fraud_words = [feature_names[i] for i in top_indices if len(feature_names[i]) > 1]

AUTO_REPLACEMENTS = {}
match_count = 0

for word in top_fraud_words:
    if word in wv:
        similars = wv.most_similar(word, topn=10)
        # 门槛调低到 0.55 以覆盖更多欺诈词汇
        valid = [s[0] for s in similars if s[1] > 0.55 and s[0] != word and len(s[0]) > 1]
        if valid:
            AUTO_REPLACEMENTS[word] = valid
            match_count += 1
            if match_count <= 5:
                print(f"  成功匹配示例: [{word}] -> {valid[:2]}")

print(f">>> 词表提取结束，共找到 {match_count} 个可替换语义词。")


# ==================== 5. 核心攻击逻辑 ====================

def attack_atgsl(text):
    original_score = get_score(text)
    if original_score < 0: return text, False, 0

    words = jieba.lcut(str(text))
    current_words = words.copy()
    current_score = original_score

    # 获取词重要度排序
    saliency = []
    for i, w in enumerate(words):
        weight = word_weight_map.get(w, 0)
        if weight > 0:
            saliency.append((i, w, weight))
    saliency = sorted(saliency, key=lambda x: x[2], reverse=True)

    modified_count = 0
    for idx, target_w, _ in saliency:
        if modified_count >= 5: break

        # --- 策略 A: 语义同义词 (第一优先级) ---
        candidates = AUTO_REPLACEMENTS.get(target_w, [])

        # 模糊匹配：如果 '资金安全' 没找到，尝试找 '资金'
        if not candidates:
            for k, v in AUTO_REPLACEMENTS.items():
                if k in target_w:
                    candidates = v
                    break

        # --- 策略 B: 特征干扰/加空格 (保底优先级) ---
        # 只有在没有同义词的情况下才考虑加空格
        if not candidates and len(target_w) > 1:
            candidates = [target_w[0] + " " + target_w[1:]]

        best_cand = None
        max_drop = -999

        for cand in candidates:
            temp_words = current_words.copy()
            temp_words[idx] = cand
            new_text = "".join(temp_words)
            new_score = get_score(new_text)

            drop = current_score - new_score
            if drop > max_drop:
                max_drop = drop
                best_cand = cand

        if best_cand:
            current_words[idx] = best_cand
            current_score -= max_drop
            modified_count += 1
            # 成功翻转则退出
            if current_score < 0:
                return "".join(current_words), True, modified_count

    return "".join(current_words), current_score < 0, modified_count


# ==================== 6. 执行实验并保存 ====================
if __name__ == "__main__":
    print(f">>> [3/4] 正在读取数据集: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH).dropna(subset=[TEXT_COL])
    texts = df[TEXT_COL].astype(str).tolist()

    print(f">>> [4/4] 开始执行攻击实验 (测试前300条)...")
    results = []
    success_count = 0
    fraud_count = 0

    for t in tqdm(texts[:300]):
        score = get_score(t)
        if score > 0:
            fraud_count += 1
            adv_text, success, changes = attack_atgsl(t)
            if success:
                success_count += 1
                results.append({
                    "original_content": t,
                    "adversarial_content": adv_text,
                    "modification_steps": changes,
                    "score_drop": score - get_score(adv_text)
                })

    print("\n" + "=" * 40)
    print(f"原本识别为欺诈样本数: {fraud_count}")
    print(f"攻击成功样本数: {success_count}")
    if fraud_count > 0:
        print(f"攻击成功率 (ASR): {success_count / fraud_count:.2%}")
    print("=" * 40)

    if results:
        output_file = os.path.join(SAVE_DIR, "attack_results_atgsl.csv")
        pd.DataFrame(results).to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"结果已保存至: {output_file}")