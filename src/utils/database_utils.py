import os
import numpy as np
import cv2
import json

# convert images to numpy embeddings database
def convert_imgs_to_db(img_dir, db_dir, swinface):
    files = os.listdir(img_dir)
    files = [f for f in files if f.endswith(".jpg") or f.endswith(".png")]

    image_db = []
    embeddings_db = []
    metadata = {}
    img_db_path = os.path.join(db_dir, "img_db.npy")
    embeddings_db_path = os.path.join(db_dir, "embeddings_db.npy")
    metadata_path = os.path.join(db_dir, "metadata.txt")
    for f in files:
        img = cv2.imread(os.path.join(img_dir, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        facial_embedding = swinface.forward(img)
        embeddings_db.append(facial_embedding)
        image_db.append(img)
        id = files.index(f)
        info = {'name': f}
        metadata[str(id)] = info
    np.save(embeddings_db_path, np.array(embeddings_db))
    np.save(img_db_path, np.array(image_db))
    # save json file
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata, outfile)
    return image_db, embeddings_db, metadata
    

def load_db(db_dir):
    image_db_path = os.path.join(db_dir, "img_db.npy")
    embeddings_db_path = os.path.join(db_dir, "embeddings_db.npy")
    metadata_path = os.path.join(db_dir, "metadata.txt")
    image_db = np.load(image_db_path, allow_pickle=True)
    embbedings_db = np.load(embeddings_db_path, allow_pickle=True)
    with open(metadata_path) as json_file:
        metadata = json.load(json_file)
    return image_db, embbedings_db, metadata