# image_search_cifar_clip.py
# install required pacakages
# ! pip install transformers datasets faiss-gpu faiss-cpu
import torch
import transformers
import datasets
import numpy as np
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

# Load CLIP model and processor
model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype = torch.float16)
processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.to('cuda')
model.eval()

# Load CIFAR-100 train dataset and extract CLIP image features 
dataset = datasets.load_dataset("cifar100", split="train")
dataset = dataset.shuffle(seed = 42)

# Function to get CLIP image features
def image2vec(examples):
    inputs = processor(images = examples['img'], return_tensors = 'pt') 
    inputs.to("cuda")
    image_features = model.get_image_features(**inputs)
    examples['image_features'] = image_features
    return examples

# Map image2vec over the dataset
dataset = dataset.map(image2vec, batched=True, batch_size=256)

# save dataset to disk for future use
# dataset.save_to_disk("./data_dir/dataset_cifar_emb.hf")

# load dataset from disk
# dataset = load_from_disk("./data_dir/dataset_cifar_emb.hf")

# hf dataset lets us add an faiss index which could be directly used for nearest neighbours search
dataset.add_faiss_index("imagevec")

# function to get nearest 50 matches for any given image
def hits_all(input_idx, dataset = dataset):

    input_embedding = np.array(dataset['imagevec'][input_idx])
    scores, retrived_examples = dataset.get_nearest_examples('imagevec',\
                                                    input_embedding, k =50)
    
    return input_idx, retrived_examples['coarse_label'], retrived_examples['fine_label'], scores



## randomly select 100 images and then find their closest image neighbours using faiss index in dataset
input_idx_list = np.random.choice(np.arange(0,len(dataset)), size = 100, replace = False)

coarse_matches = []
fine_matches = []
scores_collected = []
for i in tqdm(input_idx_list):
    _ , a, b, c = hits_all(i)
    coarse_matches.append(a)
    fine_matches.append(b)
    scores_collected.append(c)


faiss_results_cifar_clip = pd.DataFrame({"input_idx": input_idx_list, \
                                          "coarse_matches":coarse_matches,\
                                          "fine_matches": fine_matches,\
                                          "scores": scores_collected})


faiss_results_cifar_clip['input_coarse_label'] = dataset[input_idx_list]['coarse_label']
faiss_results_cifar_clip['input_fine_label'] = dataset[input_idx_list]['fine_label']


## you can check mean average precision with the following scripts, 
## mean average precision indicates in a closest K matches, how many on average are correct.

def calc_map(x, i =1):
    input_c = x['input_coarse_label']
    input_f = x['input_fine_label']

    c_matches = x['coarse_matches'][:i]
    f_matches = x['fine_matches'][:1]

    sum_c = sum([1 if x == input_c else 0 for x in c_matches])
    sum_f = sum([1 if x == input_f else 0 for x in f_matches])

    return (sum_c/i, sum_f/i)


hits_at = [1,2,3,5,10,25]
for i in hits_at:
    col_name = f"map@{i}"
    faiss_results_cifar_clip[col_name] = faiss_results_cifar_clip.apply(lambda x: calc_map(x, i = i), axis = 1)

