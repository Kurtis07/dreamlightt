from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg') # GUI 백엔드가 없는 환경에서 Matplotlib 사용 설정
import matplotlib.pyplot as plt
import io
import base64

# 기존 분석 코드를 저장한 파일에서 함수들을 가져옴
from analysis_module import analyze_nuances, visualize_embeddings_3d

app = Flask(__name__)
CORS(app) # 다른 출처(HTML 파일)의 요청을 허용

# 시각화 함수를 웹에 맞게 수정
def visualize_for_web(embeddings, labels):
    """Matplotlib 플롯을 생성하고 Base64 이미지 문자열로 변환합니다."""
    pca = plt.PCA(n_components=3)
    reduced = pca.fit_transform(embeddings)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']

    for i, (point, label) in enumerate(zip(reduced, labels)):
        ax.scatter(point[0], point[1], point[2], color=colors[label % len(colors)], alpha=0.6)
    
    ax.set_title("3D Visualization of Word Usages")
    
    # 이미지를 메모리 버퍼에 저장
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig) # 메모리에서 플롯 닫기
    
    # Base64로 인코딩하여 HTML에서 바로 사용할 수 있는 형태로 만듦
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{data}"


@app.route('/analyze', methods=['POST'])
def analyze_word():
    data = request.get_json()
    if not data or 'word' not in data:
        return jsonify({"error": "단어가 전송되지 않았습니다."}), 400

    word = data['word']
    
    # 기존 analyze_nuances 함수를 호출하되, 웹에 맞게 수정
    try:
        # analyze_nuances 함수가 결과를 반환하도록 수정했다고 가정
        # (시각화 호출 부분은 제외하고 결과 딕셔너리와 임베딩, 레이블을 반환)
        results, embeddings, labels = analyze_nuances_for_web(word)

        if results is None:
             return jsonify({"error": "분석에 필요한 문장이 부족합니다."})

        # 시각화 실행 및 Base64 인코딩된 이미지 받기
        plot_image_data = visualize_for_web(embeddings, labels)

        response_data = {
            "clusters": results,
            "plot": plot_image_data
        }
        return jsonify(response_data)

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "서버 분석 중 오류가 발생했습니다."}), 500

# analyze_nuances 함수를 웹 반환용으로 약간 수정
def analyze_nuances_for_web(target_word: str, num_clusters=3, max_sentences=500):
    # 이 함수는 원래 제공해주신 analyze_nuances 함수와 거의 동일합니다.
    # 마지막 visualize_embeddings_3d 호출 부분만 제거하고,
    # 임베딩(embeds)과 레이블(kmeans.labels_)을 결과와 함께 반환하도록 수정해야 합니다.
    
    # 아래는 analyze_nuances의 핵심 로직 (가정)
    from analysis_module import find_example_sentences, get_contextual_embeddings, get_cluster_representatives_contextual, KMeans, defaultdict
    from analysis_module import BertTokenizer, BertModel

    sents = find_example_sentences(target_word, max_sentences)
    if len(sents) < num_clusters: return None, None, None

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").eval()

    embeds, valid_sents = get_contextual_embeddings(target_word, sents, tokenizer, model)
    if len(valid_sents) < num_clusters: return None, None, None
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    kmeans.fit(embeds)

    clusters = defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(valid_sents[i])

    results = {}
    for cid, sents in clusters.items():
        label_word = get_cluster_representatives_contextual(sents, target_word, tokenizer, model)
        results[cid] = {"label": label_word, "examples": sents[:5], "count": len(sents)}

    return results, embeds, kmeans.labels_


if __name__ == '__main__':
    app.run(debug=True)