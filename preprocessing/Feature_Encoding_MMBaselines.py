#  Text/Image Feature Extraction

import os
import numpy as np
import pandas as pd

os.chdir('/nfs/user/MMRec/Sports14')
os.getcwd()

## Load text data

i_id, desc_str = 'itemID', 'description'

file_path = './'
file_name = 'meta-sports14.csv'

meta_file = os.path.join(file_path, file_name)

df = pd.read_csv(meta_file)
df.sort_values(by=[i_id], inplace=True)

print('data loaded!')
print(f'shape: {df.shape}')

df[:3]


# sentences: title + brand + category + description | All have title + description

title_na_df = df[df['title'].isnull()]
print(title_na_df.shape)

desc_na_df = df[df['description'].isnull()]
print(desc_na_df.shape)

na_df = df[df['description'].isnull() & df['title'].isnull()]
print(na_df.shape)

na3_df = df[df['description'].isnull() & df['title'].isnull() & df['brand'].isnull()]
print(na3_df.shape)

na4_df = df[df['description'].isnull() & df['title'].isnull() & df['brand'].isnull() & df['categories'].isnull()]
print(na4_df.shape)


df[desc_str] = df[desc_str].fillna(" ")
df['title'] = df['title'].fillna(" ")
df['brand'] = df['brand'].fillna(" ")
df['categories'] = df['categories'].fillna(" ")


sentences = []
for i, row in df.iterrows():
    sen = row['title'] + ' ' + row['brand'] + ' '
    cates = eval(row['categories'])
    if isinstance(cates, list):
        for c in cates[0]:
            sen = sen + c + ' '
    sen += row[desc_str]
    sen = sen.replace('\n', ' ')

    sentences.append(sen)

# sentences[:10]
print(type(sentences))
print(len(sentences))
print(sentences[:10])

# item_id
item_id = df[i_id].tolist()
print(len(item_id))

#make a dict item_id & text
dict_iid_text = dict(zip(item_id,sentences))
print(dict_iid_text[0])

import json
jsons = json.dumps(dict_iid_text)
with open("raw_text.json",'w') as f:
    f.write(jsons)
print('raw text json done')

# test = []
# for line in open("raw_text.json").readlines():
#     test.append(json.loads(line))
# print('raw text json:', test[0])

# import json
# with open("raw_text.json",'w') as f:
#     for i in range(0,len(sentences)):
#         json.dump(sentences[i],f)
# print('raw text done')

# jsons = []
# for line in open("raw_text.json").readlines():
#     jsons.append(json.loads(line))
# print('raw text json:', jsons[0])
        


course_list = df[i_id].tolist()
#sentences = df[desc_str].tolist()

assert course_list[-1] == len(course_list) - 1

# should `pip install sentence_transformers` first
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence_embeddings = model.encode(sentences)
print('text encoded!')

assert sentence_embeddings.shape[0] == df.shape[0]
np.save(os.path.join(file_path, 'text_feat.npy'), sentence_embeddings)
print('done!')


sentence_embeddings[:10]

load_txt_feat = np.load('text_feat.npy', allow_pickle=True)
print(load_txt_feat.shape)
load_txt_feat[:10]



# Image encoder (V0)，following LATTICE, averaging over for missed items

df[:5]

import array

def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    asin = f.read(10).decode('UTF-8')
    if asin == '': break
    a = array.array('f')
    a.fromfile(f, 4096)
    yield asin, a.tolist()


img_data = readImageFeatures("image_features_Sports_and_Outdoors.b")
item2id = dict(zip(df['asin'], df['itemID']))

feats = {}
avg = []
for d in img_data:
    if d[0] in item2id:
        feats[int(item2id[d[0]])] = d[1]
        avg.append(d[1])
avg = np.array(avg).mean(0).tolist()

ret = []
non_no = []
for i in range(len(item2id)):
    if i in feats:
        ret.append(feats[i])
    else:
        non_no.append(i)
        ret.append(avg)

print('# of items not in processed image features:', len(non_no))
assert len(ret) == len(item2id)
np.save('image_feat.npy', np.array(ret))
np.savetxt("missed_img_itemIDs.csv", non_no, delimiter =",", fmt ='%d')
print('done!')
