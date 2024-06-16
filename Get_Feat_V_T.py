import os
import numpy as np
from sentence_transformers import SentenceTransformer

#init model
model = SentenceTransformer('all-MiniLM-L6-v2')

# type(sentences) = list
sentence_embeddings = model.encode(sentences)
np.save(os.path.join(file_path, 'text_feat.npy'), sentence_embeddings)

#view embedding
load_txt_feat = np.load('text_feat.npy', allow_pickle=True)
print(load_txt_feat.shape)
print(load_txt_feat[:10])
