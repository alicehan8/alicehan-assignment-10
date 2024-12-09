import os
import torch
#import torchvision.transforms as transforms
from PIL import Image
#from open_clip import create_model_and_transforms, tokenizer
import open_clip
import torch.nn.functional as F
# import pandas as pd
from tqdm import tqdm
import numpy as np
import io
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def find_image(df, query_embedding):
    # keep an array of images and similarities so I can get top 5 later
    impaths = []
    similarities = []

    # Ensure query_embedding is a PyTorch tensor and has the correct dimensions
    # query_embedding = torch.tensor(query_embedding).unsqueeze(0) if query_embedding.ndim == 1 else torch.tensor(query_embedding)
    # trying this instead
    query_embedding = query_embedding.clone().detach()
    query_embedding = query_embedding.unsqueeze(0) if query_embedding.ndim == 1 else query_embedding

    for _, row in df.iterrows():
        # Check if the embedding exists and is valid
        if not isinstance(row['embedding'], (list, torch.Tensor, np.ndarray)):
            print(f"Invalid embedding at row {_}: {row['embedding']}")
            continue
        
        # Convert the dataset embedding to a PyTorch tensor
        dataset_embedding = torch.tensor(row['embedding'], dtype=torch.float32)
        
        # Ensure correct dimensions for cosine similarity
        dataset_embedding = dataset_embedding.unsqueeze(0) if dataset_embedding.ndim == 1 else dataset_embedding

        # Compute cosine similarity
        similarity = F.cosine_similarity(query_embedding, dataset_embedding).item()

        impaths.append(row['file_name'])
        similarities.append(similarity)

    # fix image paths
    top_5 = np.argsort(similarities)[::-1][:5]

    top_5_images = [impaths[i] for i in top_5]
    top_5_sims = [similarities[i] for i in top_5]

    return top_5_images, top_5_sims


def embed_image(image_path):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32-quickgelu', pretrained='openai') # adding -quickgelu
    # This converts the image to a tensor
    if isinstance(image_path, str):
        image = preprocess(Image.open(image_path)).unsqueeze(0)
    else:
        image = Image.open(io.BytesIO(image_path.read())) 
        image = preprocess(image).unsqueeze(0)

    # This calculates the query embedding
    query_embedding = F.normalize(model.encode_image(image))
    return query_embedding

def embed_text(text_query):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32-quickgelu', pretrained='openai') # wait do i want image and text to u se the same model tho
    token = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    text = token([text_query]) # change this to be what you want...
    query_embedding = F.normalize(model.encode_text(text))
    return query_embedding

def embed_hybrid(image_path, text_query, weight):
    image_query = embed_image(image_path)# F.normalize(model.encode_image(image))
    text_query = embed_text(text_query)#F.normalize(model.encode_text(text))

    lam  = weight # tune this

    query_embedding = F.normalize(lam * text_query + (1.0 - lam) * image_query)
    return query_embedding

# Already completed for you
def load_images(image_dir, max_images=None, target_size=(224, 224)):
    images = []
    image_names = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_dir, filename))
            img = img.convert('L')  # Convert to grayscale ('L' mode)
            img = img.resize(target_size)  # Resize to target size
            img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array.flatten())  # Flatten to 1D
            image_names.append(filename)
        if max_images and i + 1 >= max_images:
            break
    return np.array(images), image_names

def find_image_pca(df, k, image_path):
    impaths = []
    distances = []

    # Directory containing images
    image_dir = "results"  # Your folder path

    # Train PCA
    train_images, train_image_names = load_images(image_dir, max_images=2000, target_size=(224, 224))
    print(f"Loaded {len(train_images)} images for PCA training.")

    # Apply PCA
    pca = PCA(k)  # Initialize PCA with k components
    pca.fit(train_images)
    print(f"Trained PCA on {len(train_images)} samples.")

    # Transform images
    transform_images, transform_image_names = load_images(image_dir, max_images=10000, target_size=(224, 224))
    print(f"Loaded {len(transform_images)} images for transformation.")
    reduced_embeddings = pca.transform(transform_images)  # Transform all images
    print(f"Reduced embeddings for {len(transform_images)} images.")

    # Prepare query embedding
    image = Image.open(io.BytesIO(image_path.read())) 
    image = image.convert('L')  # Convert to grayscale ('L' mode)
    image = image.resize((224, 224))  # Resize to target size
    img_array = np.asarray(image, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
    image_flat = img_array.flatten()  # Flatten to 1D
    query_embedding = pca.transform(np.array(image_flat).reshape(1, -1))

    # Find nearest neighbors
    top_indices, top_distances = nearest_neighbors(query_embedding, reduced_embeddings)
    print("Top indices:", top_indices)
    print("Top distances:", top_distances)

    for i, index in enumerate(top_indices):
        impath = df['file_name'].iloc[index]
        similarity = top_distances[i].item()
        impaths.append(impath)
        distances.append(similarity)

    return impaths, distances


def nearest_neighbors(query_embedding, embeddings, top_k=7):
    # query_embedding: The embedding of the query item (e.g., the query image) in the same dimensional space as the other embeddings.
    # embeddings: The dataset of embeddings that you want to search through for the nearest neighbors.
    # top_k: The number of most similar items (nearest neighbors) to return from the dataset.
    # Hint: flatten the "distances" array for convenience because its size would be (1,N)
    distances = euclidean_distances(query_embedding, embeddings).flatten() #Use euclidean distance
    nearest_indices = np.argsort(distances)[:top_k] #get the indices of ntop k results
    return nearest_indices, distances[nearest_indices]
