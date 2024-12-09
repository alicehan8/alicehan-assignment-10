from flask import Flask, render_template, request, jsonify, send_from_directory
import os
# from neural_networks import visualize
from image_search import find_image, embed_image, embed_text, embed_hybrid, find_image_pca
from PIL import Image
import pandas as pd
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# is there where i define df
df = pd.read_pickle('image_embeddings.pickle')

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle experiment parameters and trigger the experiment
@app.route('/image_search', methods=['POST'])
def image_search():
    query_type = request.form['query_type']
    embedding_type = request.form.get('embedding_type', 'clip')
    use_pca = embedding_type == 'pca'
    k = int(request.form.get('k', 10))

    if query_type == "text":
        text_query = request.form['text_query']
        # Get the query embedding for the text
        query_embedding = embed_text(text_query)
        images, max_similarities = find_image(df, query_embedding)  # df is your dataset with embeddings
    elif query_type == "image":
        image_file = request.files['image_query']
        # image_name = secure_filename(image_file.filename)  # You might need to save the image or use a temporary path
        # #filename = secure_filename(image_file.filename)
        # image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
        # image_file.save(image_path)
        # Get the query embedding for the image
        if not use_pca:
            query_embedding = embed_image(image_file) # or image_path
            images, max_similarities = find_image(df, query_embedding)
        else:
            #query_embedding = pca_image(image_file)
            images, max_similarities = find_image_pca(df, k, image_file)  # df is your dataset with embeddings
    elif query_type == "hybrid":
        text_query = request.form['text_query']
        weight = float(request.form['weight'])
        image_file = request.files['image_query']
        #image_path = image_file.filename  # Similarly, save or handle the image path
        # Get the query embeddings for both image and text -> dont need these anyore since embed hybrid handles it
        # image_query_embedding = embed_image(image_path)
        # text_query_embedding = embed_text(text_query)
        # Combine the embeddings using the hybrid method
        query_embedding = embed_hybrid(image_file, text_query, weight)
        images, max_similarities = find_image(df, query_embedding)  # df is your dataset with embeddings
    
    # results = [
    #     {
    #         "image_url": f"/results/{os.path.basename(images)}",
    #         "similarity_score": max_similarity,
    #     }
    # ]   
    return jsonify({"images": images, "top_sims": max_similarities})

# Route to serve result images
@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True)