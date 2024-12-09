# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from flask import Flask, request, render_template, send_from_directory
import open_clip

top_k = 5
df = pd.read_pickle('image_embeddings.pickle')
all_embeddings = np.stack(df['embedding'].values)  # [num_images, embedding_dim]
image_folder = "coco_images_resized"


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')


all_embeddings_tensor = torch.tensor(all_embeddings, device=device)


pca_dim = 300
pca = PCA(n_components=pca_dim)
all_embeddings_pca = pca.fit_transform(all_embeddings)  # shape: [num_images, pca_dim]
all_embeddings_pca_tensor = torch.tensor(all_embeddings_pca, device=device)



def predict_using_pretrained(image_path = None, text=None, lam=0.8,top_k = 5):
    print(image_path)
    if not image_path:
        if text is not None and text.strip() != "":
            text_token = tokenizer([text])
            text_embedding = F.normalize(model.encode_text(text_token.to(device)), p=2, dim=1)
        cosine_similarities = (text_embedding@ all_embeddings_tensor.T).squeeze(0)
        top_values, top_indices = torch.topk(cosine_similarities, k=top_k)
        top_indices = top_indices.cpu().numpy().tolist()
        result_files = [df.iloc[idx]['file_name'] for idx in top_indices]
        return result_files
    query_img = Image.open(image_path).convert("RGB")
    query_img_tensor = preprocess_val(query_img).unsqueeze(0).to(device)
    query_img_embedding = F.normalize(model.encode_image(query_img_tensor), p=2, dim=1)

    if text is not None and text.strip() != "":
        text_token = tokenizer([text])
        text_embedding = F.normalize(model.encode_text(text_token.to(device)), p=2, dim=1)
        query_embedding = F.normalize(lam * text_embedding + (1 - lam) * query_img_embedding, p=2, dim=1)
    else:
        query_embedding = query_img_embedding

    cosine_similarities = (query_embedding @ all_embeddings_tensor.T).squeeze(0)
    top_values, top_indices = torch.topk(cosine_similarities, k=top_k)
    top_indices = top_indices.cpu().numpy().tolist()

    result_files = [df.iloc[idx]['file_name'] for idx in top_indices]
    return result_files


def predict_using_pca(image_path,top_k = 5):
    query_image = Image.open(image_path).convert("RGB")  # your query image
    query_image_tensor = preprocess_val(query_image).unsqueeze(0).to(device)  # [1,3,H,W]
    query_embedding = model.encode_image(query_image_tensor)  # [1, embedding_dim]
    query_embedding = F.normalize(query_embedding, p=2, dim=1).cpu()

    query_embedding_pca = pca.transform(query_embedding.detach().numpy())[:, :pca_dim]  # shape: [1, k]
    db_embeddings = torch.tensor(all_embeddings_pca)  # [num_images, k]
    query_vector = torch.tensor(query_embedding_pca)  # [1, k]t

    cosine_similarities = (query_vector @ db_embeddings.T).squeeze(0)  # shape: [num_images]
    top_values, top_indices = torch.topk(cosine_similarities, k=top_k)
    top_indices = top_indices.cpu().numpy().tolist()

    result_files = [df.iloc[idx]['file_name'] for idx in top_indices]
    return result_files


app = Flask(__name__)
upload_path = os.path.join("static", "uploads")
os.makedirs(upload_path, exist_ok=True)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    method = request.form.get('method', 'pretrained')
    query_text = request.form.get('query_text', '')
    lam_str = request.form.get('lam', '0.8')
    try:
        lam = float(lam_str)
    except:
        lam = 0.8

    # 获取上传的图片
    # if 'query_image' not in request.files:
    #     return "未找到上传的图片文件", 400
    # file = request.files['query_image']
    # if file.filename == '':
    #     return "未选择文件", 400

    # 保存上传图片到uploads目录
    file = request.files['query_image']
    print(file)
    query_image_path = ""
    if file:
        file = request.files['query_image']
        query_image_path = os.path.join(upload_path, file.filename)
        file.save(query_image_path)
    top_k = 5
    if method == 'pretrained':
        result_files = predict_using_pretrained(query_image_path, text=query_text, lam=lam, top_k=top_k)
    else:
        result_files = predict_using_pca(query_image_path, top_k=top_k)

    return render_template('index.html', result_images=result_files)


@app.route('/coco_images_resized/<path:filename>')
def show_image(filename):
    return send_from_directory('coco_images_resized', filename)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
