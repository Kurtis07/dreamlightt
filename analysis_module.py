import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from datasets import load_dataset
import numpy as np
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if 5 < len(s.strip().split()) < 40]


def find_example_sentences(target_word: str, max_sentences: int = 500):
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]")
    print(f"🔍 '{target_word}' 관련 문장 수집 중...")
    sentences = []
    for item in dataset:
        text = item['text']
        for sent in split_into_sentences(text):
            if target_word.lower() in sent.lower():
                sentences.append(sent)
                if len(sentences) >= max_sentences:
                    return sentences
    return sentences


def get_contextual_embeddings(target_word, sentences, tokenizer, model):
    embeddings = []
    valid_sentences = []
    print(f"🧠 '{target_word}' 임베딩 추출 중...")
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512)
        tokens = tokenizer.tokenize(sent)
        target_tokens = tokenizer.tokenize(target_word)

        match_idx = -1
        for i in range(len(tokens) - len(target_tokens) + 1):
            if tokens[i:i + len(target_tokens)] == target_tokens:
                match_idx = i + 1  # [CLS] 토큰 고려
                break
        if match_idx == -1:
            continue

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        last4 = torch.stack(hidden_states[-4:])  # 마지막 4개 레이어
        avg_vec = last4.mean(dim=0)[0, match_idx]  # 배치 0, 해당 토큰 위치
        embeddings.append(avg_vec.numpy())
        valid_sentences.append(sent)
    return np.array(embeddings), valid_sentences


def get_cluster_representatives_contextual(cluster_sentences, target_word, tokenizer, model):
    vecs = []
    word_candidates = []

    for sent in cluster_sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512)
        tokens = tokenizer.tokenize(sent)
        target_tokens = tokenizer.tokenize(target_word)

        match_idx = -1
        for i in range(len(tokens) - len(target_tokens) + 1):
            if tokens[i:i + len(target_tokens)] == target_tokens:
                match_idx = i + 1
                break
        if match_idx == -1:
            continue

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        last4 = torch.stack(hidden_states[-4:])
        vec = last4.mean(dim=0)[0, match_idx]
        vecs.append(vec.numpy())

        # 문장에서 알파벳 단어 후보 추출 (3글자 이상)
        word_list = re.findall(r'\b[a-zA-Z]+\b', sent)
        for word in word_list:
            if word.lower() != target_word.lower() and len(word) > 3:
                word_candidates.append(word.lower())

    if not vecs:
        return "UNKNOWN"

    mean_vec = np.mean(vecs, axis=0)
    mean_vec /= np.linalg.norm(mean_vec)

    # 후보 단어 상위 20개 뽑기
    top_candidates = [w for w, _ in Counter(word_candidates).most_common(20)]
    candidate_vecs = []
    candidate_words = []

    for word in top_candidates:
        inputs = tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            word_embed = model.embeddings.word_embeddings(inputs['input_ids'])[0, 1]
        word_vec = word_embed.numpy()
        word_vec /= np.linalg.norm(word_vec)
        candidate_vecs.append(word_vec)
        candidate_words.append(word)

    sims = np.dot(candidate_vecs, mean_vec)
    best_idx = np.argmax(sims)

    return candidate_words[best_idx] if candidate_words else "UNKNOWN"


def visualize_embeddings_3d(embeddings, labels):
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(embeddings)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']

    for i, (point, label) in enumerate(zip(reduced, labels)):
        ax.scatter(point[0], point[1], point[2], color=colors[label % len(colors)], alpha=0.6)

    ax.set_title("3D Visualization of Word Usages")
    plt.show()


def analyze_nuances(target_word: str, num_clusters=3, max_sentences=500):
    sents = find_example_sentences(target_word, max_sentences)
    if len(sents) < num_clusters:
        return None

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").eval()

    embeds, valid_sents = get_contextual_embeddings(target_word, sents, tokenizer, model)
    if len(valid_sents) < num_clusters:
        return None

    print("🔗 클러스터링 중...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    kmeans.fit(embeds)

    clusters = defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(valid_sents[i])

    results = {}
    print("📌 대표 의미 추론 중...")
    for cid, sents in clusters.items():
        label_word = get_cluster_representatives_contextual(sents, target_word, tokenizer, model)
        results[cid] = {
            "label": label_word,
            "examples": sents[:5],
            "count": len(sents)
        }

    # 시각화 실행
    visualize_embeddings_3d(embeds, kmeans.labels_)

    return results


if __name__ == "__main__":
    word = input("분석할 영어 단어 입력 (예: spring, bank, light): ").strip()
    result = analyze_nuances(word)

    if result is None:
        print("❌ 오류: 문장 수 부족 혹은 유효 문장 부족")
    else:
        print(f"\nBERT 기반 '{word}' 뉘앙스 자동 분석 결과:")
        print("=" * 60)
        for cid in sorted(result.keys()):
            label = result[cid]["label"]
            examples = result[cid]["examples"]
            total = result[cid]["count"]
            print(f"\n뉘앙스 #{cid + 1} ― 대표 단어: {label}")
            for ex in examples:
                print(f'  • "{ex}"')
            if total > len(examples):
                print(f"  … 등 {total - len(examples)}개 문장 더 있음")
