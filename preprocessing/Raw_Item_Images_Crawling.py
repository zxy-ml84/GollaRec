import os
import numpy as np
import pandas as pd
import json

os.chdir('/nfs/user/MMRec/Amazon_Clothing_Dataset')
os.getcwd()

i_id, desc_str = 'itemID', 'description'

file_path = './'
file_name = 'meta-clothing.csv'

meta_file = os.path.join(file_path, file_name)

df = pd.read_csv(meta_file)
df.sort_values(by=[i_id], inplace=True)

print('data loaded!')
print(f'shape: {df.shape}')

df[:3]

item_ids = df[i_id].tolist()
image_urls = df['imUrl'].tolist()
print(type(image_urls[126]))

import requests
import math
import time

for i in range(len(item_ids)):
    item_id = item_ids[i]
    image_url = image_urls[i]
    item_2_deleted = []
    if isinstance(image_url, float) and math.isnan(image_url):
        print(f"Skipping item {item_id} due to NaN image URL")
        item_2_deleted.append(item_id)
        continue
    retry_count = 0
    while retry_count < 3:
        response = requests.get(image_url)
        if response.status_code == 200:
#             file_path = os.path.join("images", f"{item_id}.jpg")
            with open(f"/nfs/user/MMRec/Amazon_Clothing_Dataset/raw_image_id/{item_id}.jpg", "wb") as f:
                f.write(response.content)
                print(f"Downloaded image for item {item_id}")
            break
        elif response.status_code == 404:
            print(f"Skipping item {item_id} because image URL is not found: {image_url}")
            item_2_deleted.append(item_id)
            break
        else:
            print(f"Failed to download image for item {item_id}. Status code: {response.status_code}")
            retry_count += 1
            time.sleep(1) # Wait for 1 second before retrying
    if retry_count == 3:
        print(f"Giving up on item {item_id} after 3 download attempts")
print('Downloading finish')
print('item2delete',item_2_deleted)
with open('item2delete.txt', 'w') as file:
    for item in item_2_deleted:
        file.write(str(item) + '\n')

# import requests
# import math
# import time
# import logging

# logging.basicConfig(filename='download_images.log', level=logging.INFO)

# item_2_deleted = []

# for i in range(len(item_ids)):
#     item_id = item_ids[i]
#     image_url = image_urls[i]
    
#     if isinstance(image_url, float) and math.isnan(image_url):
#         logging.warning(f"Skipping item {item_id} due to NaN image URL")
#         continue
        
#     retry_count = 0
#     while retry_count < 3:
#         try:
#             response = requests.get(image_url)
#             if response.status_code == 200:
#                 with open(f"/nfs/user/MMRec/Amazon_Sports_Dataset/raw_image_id/{item_id}.jpg", "wb") as f:
#                     f.write(response.content)
#                     logging.info(f"Downloaded image for item {item_id}")
#                 break
#             elif response.status_code == 404:
#                 logging.warning(f"Skipping item {item_id} because image URL is not found: {image_url}")
#                 item_2_deleted.append(item_id)
#                 break
#             else:
#                 logging.warning(f"Failed to download image for item {item_id}. Status code: {response.status_code}")
#                 retry_count += 1
#                 time.sleep(1) # Wait for 1 second before retrying
#         except Exception as e:
#             logging.error(f"Failed to download image for item {item_id}. Exception: {e}")
#             retry_count += 1
#             time.sleep(1) # Wait for 1 second before retrying
    
#     if retry_count == 3:
#         logging.warning(f"Giving up on item {item_id} after 3 download attempts")
    
# logging.info('Downloading finish')
# logging.info(f'item2delete: {item_2_deleted}')

# with open('item2delete.txt', 'w') as file:
#     for item in item_2_deleted:
#         file.write(str(item) + '\n')


import os

folder_path = "/nfs/user/MMRec/Amazon_Clothing_Dataset/raw_image_id"
num_files = len(os.listdir(folder_path))
print(f"There are {num_files} files in {folder_path}")

import os

folder_path = "/nfs/user/MMRec/Amazon_Clothing_Dataset/raw_image_id" 
start_id = 0
end_id = 23032

missing_ids = []

for i in range(start_id, end_id+1):
    file_path = os.path.join(folder_path, str(i) + ".jpg")
    if not os.path.exists(file_path):
        missing_ids.append(i)

print("Missing IDs:", missing_ids)
print("Num of missing:", len(missing_ids))


rslt_file = 'clothing-indexed-v4.inter'
df = pd.read_csv(rslt_file, sep='\t')
print(f'shape: {df.shape}')
df[:10]

import pandas as pd

rslt_file = '/nfs/user/MMRec/Amazon_Clothing_Dataset/clothing-indexed-v4.inter'
df = pd.read_csv(rslt_file, sep='\t')
# missing_ids = [126, 328, 735, 836, 837, 839, 840, 841, 843, 1027, 1252, 1267, 1908, 1926, 2124, 2740, 3144, 3737, 5143, 5849, 6282, 6714, 7043, 7207, 7826, 7864, 8034, 8136, 8528, 9085, 9315, 9557, 9989, 10307, 10326, 10340, 10375, 10381, 10720, 10754, 10952, 10955, 11053, 11750, 11765, 12251, 12390, 12698, 12713, 13340, 13406, 13433, 13553, 13819, 13863, 13910, 14486, 14643, 14999, 15329, 15579, 16057, 16734, 17010, 17159, 17632, 17633, 17635, 17813, 17872]
# missing_ids = [284, 600, 967, 1255, 1282, 2046, 2480, 3790, 4123, 4365, 5209, 5303, 5328]
# missing_ids = [153, 1690, 3469, 3550, 5409, 5636, 6106, 8391, 8883, 9071, 9722, 10340, 10395, 10397, 10500, 10777, 10804, 10873, 11392, 11491, 11567, 11572, 11733, 11857, 12322, 20105, 20201, 21267, 22741, 26863, 28383, 28460, 28826, 28852, 29381, 29920, 30019, 30360, 32099, 32445, 32607, 33410, 34637, 34762, 35176, 37636, 41297, 42501, 43225, 43553, 45300, 45346, 48589, 48984, 54719, 55331, 55985, 60137, 61787, 61847, 62576, 62688]
missing_ids = [42, 43, 112, 169, 172, 176, 180, 193, 216, 241, 273, 297, 423, 432, 444, 465, 483, 518, 571, 572, 575, 576, 577, 579, 580, 593, 596, 606, 647, 648, 719, 723, 728, 747, 759, 760, 767, 913, 936, 940, 955, 1187, 1226, 1248, 1266, 1403, 1416, 1469, 1472, 1483, 1494, 1523, 1586, 1622, 1660, 1669, 1702, 1721, 1737, 1741, 1824, 1901, 1903, 2029, 2037, 2077, 2078, 2128, 2142, 2144, 2148, 2153, 2168, 2218, 2269, 2273, 2281, 2380, 2393, 2461, 2492, 2507, 2508, 2509, 2556, 2579, 2580, 2628, 2737, 2755, 2761, 2771, 2810, 2826, 2827, 2848, 2859, 2892, 2926, 3048, 3075, 3113, 3114, 3123, 3174, 3263, 3291, 3318, 3324, 3335, 3347, 3386, 3387, 3390, 3393, 3394, 3395, 3412, 3426, 3429, 3430, 3433, 3547, 3553, 3581, 3604, 3809, 3834, 3938, 4041, 4235, 4269, 4363, 4412, 4561, 4564, 4600, 4663, 4719, 4720, 4727, 4735, 4747, 4795, 4822, 4823, 4827, 4934, 4938, 4956, 4970, 5008, 5050, 5160, 5218, 5225, 5261, 5284, 5297, 5354, 5374, 5491, 5505, 5597, 5687, 5752, 5821, 5869, 5959, 6024, 6027, 6146, 6394, 6465, 6520, 6607, 6629, 6633, 6634, 6635, 6636, 6698, 6699, 6763, 6771, 6774, 6791, 6911, 6937, 6950, 6959, 6969, 7016, 7043, 7080, 7086, 7087, 7088, 7089, 7111, 7132, 7175, 7177, 7202, 7244, 7266, 7284, 7285, 7286, 7518, 7644, 7714, 7780, 7805, 7870, 7881, 7938, 7953, 7955, 8042, 8279, 8281, 8303, 8360, 8516, 8582, 8624, 8657, 8715, 8775, 8839, 8840, 8867, 9040, 9205, 9206, 9232, 9280, 9317, 9540, 9714, 9784, 9984, 9996, 10026, 10030, 10033, 10076, 10106, 10179, 10238, 10241, 10278, 10340, 10497, 10787, 10788, 10907, 11312, 11333, 11334, 11464, 11528, 11532, 11538, 11578, 11613, 11629, 11639, 11666, 11775, 11862, 12078, 12123, 12187, 12354, 12654, 12656, 12753, 12997, 13067, 13182, 13201, 13203, 13246, 13563, 13576, 13684, 13867, 13875, 13892, 13903, 14019, 14103, 14259, 14381, 14535, 14799, 14806, 14833, 14927, 15105, 15302, 15344, 15353, 15354, 15387, 15527, 15528, 15747, 15834, 15918, 15927, 15958, 15960, 15963, 15965, 15978, 15984, 16032, 16226, 16308, 16312, 16335, 16345, 16351, 16467, 16480, 16489, 16496, 16507, 16534, 16611, 16641, 16650, 16786, 16833, 16834, 17086, 17165, 17635, 17679, 17691, 17743, 18162, 18360, 18404, 18426, 18507, 18548, 18685, 18809, 18888, 18934, 18937, 19005, 19008, 19096, 19132, 19139, 19141, 19152, 19161, 19174, 19176, 19312, 19454, 19512, 19612, 19665, 19761, 19859, 19872, 19873, 19874, 19880, 19896, 19897, 19917, 19919, 19921, 19931, 20075, 20096, 20197, 20381, 20445, 20446, 20451, 20468, 20471, 20492, 20509, 20544, 20561, 20665, 20726, 20727, 20729, 20730, 20740, 20746, 20755, 20756, 20795, 20796, 20797, 20808, 20914, 20942, 20969, 21091, 21107, 21117, 21145, 21192, 21200, 21201, 21205, 21241, 21279, 21296, 21324, 21335, 21359, 21369, 21370, 21386, 21408, 21409, 21410, 21411, 21425, 21441, 21442, 21445, 21447, 21449, 21450, 21453, 21454, 21486, 21494, 21528, 21565, 21576, 21591, 21593, 21595, 21640, 21662, 21663, 21664, 21665, 21666, 21732, 21771, 21791, 21793, 21807, 21808, 21809, 21833, 21834, 21835, 21851, 21933, 21950, 21961, 21980, 21993, 21994, 21995, 22020, 22029, 22055, 22067, 22106, 22107, 22118, 22119, 22120, 22150, 22151, 22166, 22174, 22175, 22176, 22184, 22194, 22198, 22203, 22229, 22233, 22246, 22252, 22262, 22294, 22314, 22315, 22316, 22317, 22318, 22325, 22326, 22332, 22340, 22355, 22374, 22376, 22377, 22378, 22397, 22402, 22492, 22500, 22506, 22507, 22508, 22509, 22510, 22552, 22573, 22599, 22638, 22642, 22686, 22688, 22715, 22762, 22763, 22873, 22874, 22951]
# Filter the DataFrame to exclude rows with missing IDs
df_filtered = df[~df['itemID'].isin(missing_ids)]

# print(f'df_filtered:',df_filtered)
print(f'shape: {df_filtered.shape}')

# Save the filtered DataFrame to a CSV file
df_filtered.to_csv('/nfs/user/MMRec/Amazon_Clothing_Dataset/clothing-indexed-v5.inter', index=False)
print('file save')

# df_filtered['itemID']
unique_item_ids = df_filtered['itemID'].unique()
unique_item_ids.sort()
new_item_ids = range(len(unique_item_ids))
item_id_map = dict(zip(unique_item_ids, new_item_ids))
new_df = df_filtered.copy()
new_df['itemID'] = df_filtered['itemID'].map(item_id_map)
item_id_mapping_df = pd.DataFrame({'old_item_id': list(item_id_map.keys()),'new_item_id': list(item_id_map.values())})

new_df.to_csv('/nfs/user/MMRec/Amazon_Clothing_Dataset/clothing-indexed-v6.inter', sep=',', index=False)
print('---- file save ----')

print(new_df)
print(max(new_df['itemID'].values))
print(item_id_mapping_df)

