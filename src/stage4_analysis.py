import joblib
import numpy as np
import pandas as pd
import jieba
import os
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from tqdm import tqdm

# ==================== 1. 路径与环境配置 ====================
MODEL_DIR = r"D:\pylearning"
FASTTEXT_PATH = r"D:\nlp_models\fasttext\cc.zh.300.vec"
DATA_PATH = r"D:\Natural Language Processing\实验1\通话数据互动策略结果\训练集结果.csv"
SAVE_DIR = r"D:\Natural Language Processing\大作业"
TEXT_COL = "specific_dialogue_content"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print(">>> [1/4] 正在初始化 SVM 和 FastText 词向量...")
clf = joblib.load(os.path.join(MODEL_DIR, "svm_clf.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
wv = KeyedVectors.load_word2vec_format(FASTTEXT_PATH, binary=False, limit=200000)

feature_names = vectorizer.get_feature_names_out()
weights = clf.coef_[0]
word_weight_map = dict(zip(feature_names, weights))


def get_score(text):
    """计算 SVM 分数辅助函数"""
    processed = " ".join(jieba.lcut(str(text)))
    return clf.decision_function(vectorizer.transform([processed]))[0]


# ==================== 2. 核心实验引擎 ====================

def run_attack_experiment(target_n, texts_to_test):
    # 动态构建基于当前 N 的替换词表
    top_indices = np.argsort(weights)[::-1][:target_n]
    top_fraud_words = {feature_names[i] for i in top_indices if len(feature_names[i]) > 1}

    current_replacements = {}
    # 为了速度，只对权重最高的 500 个词生成同义词，其他的用模糊匹配或空格
    for word in list(top_fraud_words)[:500]:
        if word in wv:
            similars = wv.most_similar(word, topn=5)
            valid = [s[0] for s in similars if s[1] > 0.45 and s[0] != word]
            if valid: current_replacements[word] = valid

    success_count = 0
    fraud_count = 0

    for t in texts_to_test:
        orig_score = get_score(t)
        if orig_score <= 0: continue

        fraud_count += 1
        words = jieba.lcut(str(t))
        curr_words = words.copy()

        # 寻找当前 N 范围内的显著词
        saliency = []
        for i, w in enumerate(words):
            if w in top_fraud_words:
                saliency.append((i, w, word_weight_map.get(w, 0)))
        saliency = sorted(saliency, key=lambda x: x[2], reverse=True)

        # 尝试最多修改 10 次 (体现深度搜索)
        for idx, target_w, _ in saliency[:10]:
            candidates = current_replacements.get(target_w, [])

            # 模糊匹配
            if not candidates:
                for k, v in current_replacements.items():
                    if k in target_w:
                        candidates = v
                        break

            # 空格保底
            if not candidates and len(target_w) > 1:
                candidates = [target_w[0] + " " + target_w[1:]]

            if not candidates: continue

            best_cand = None
            max_drop = -999
            for cand in candidates:
                temp_words = curr_words.copy()
                temp_words[idx] = cand
                new_score = get_score("".join(temp_words))
                if (orig_score - new_score) > max_drop:
                    max_drop = orig_score - new_score
                    best_cand = cand

            if best_cand:
                curr_words[idx] = best_cand
                if get_score("".join(curr_words)) < 0:
                    success_count += 1
                    break

    return (success_count / fraud_count * 100) if fraud_count > 0 else 0


# ==================== 3. 运行多梯度测试 ====================

print(">>> [2/4] 加载数据集...")
df = pd.read_csv(DATA_PATH).dropna(subset=[TEXT_COL])
test_samples = df[TEXT_COL].astype(str).tolist()[:300]  # 每次测试前300条

# 设定 N 的采样点（涵盖了数量级的跨度）
n_values = [10, 50, 100, 200, 500, 1000, 2000, 4000, 7000, 10000]
asr_results = []

print(">>> [3/4] 开始执行全量程 ASR 趋势扫描...")
for n in n_values:
    asr = run_attack_experiment(n, test_samples)
    asr_results.append(asr)
    print(f"完成节点 N = {n:5d} | ASR = {asr:.2f}%")

# ==================== 4. 绘制学术趋势图 ====================

print(">>> [4/4] 正在生成可视化图表...")
plt.figure(figsize=(12, 7))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False

# 使用对数坐标展示 N 的变化
plt.semilogx(n_values, asr_results, marker='D', markersize=8, color='#2980B9',
             linestyle='-', linewidth=2.5, label='ATGSL Attack Success Rate')

# 标注每个点的数据
for i, txt in enumerate(asr_results):
    plt.annotate(f"{txt:.1f}%", (n_values[i], asr_results[i]), xytext=(0, 10),
                 textcoords="offset points", ha='center', fontsize=11, fontweight='bold')

# 美化图表
plt.title("ATGSL 实验分析：特征空间规模(N)与攻击成功率(ASR)的关系", fontsize=16, pad=20)
plt.xlabel("目标特征词提取规模 N (Log Scale)", fontsize=12)
plt.ylabel("攻击成功率 (ASR %)", fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.axhline(y=max(asr_results), color='r', linestyle=':', alpha=0.5, label='Saturation Level')
plt.legend(loc='lower right', fontsize=12)

# 保存图片到你的大作业目录
plt.savefig(os.path.join(SAVE_DIR, "asr_trend_analysis.png"), dpi=300)
plt.show()

print(f"\n实验完成！图片已保存至: {SAVE_DIR}\\asr_trend_analysis.png")