
# Image Search using CLIP and CIFAR-100

This script demonstrates how to use CLIP image embeddings and faiss indexing to efficiently search for similar images in the CIFAR-100 dataset.


## Overview

1. Load CIFAR-100 train split and CLIP model
2. Extract CLIP image features for each image
3. Save dataset with image embeddings to disk
4. Load dataset and add a faiss index
5. Define a function to search for nearest neighbors
6. Loop through some sample images and find nearest matches
7. Save results to a DataFrame

## Requirements
* transformers
* datasets
* faiss-gpu or faiss-cpu
* pandas
* numpy
* tqdm

## Usage/Examples


The key steps are:

1. Extract CLIP image features with `image2vec` function
2. Add a faiss index to dataset with `dataset.add_faiss_index("imagevec")`
3. Search for nearest matches with `dataset.get_nearest_examples()`
4. The faiss index enables efficient similarity search on the CLIP embeddings.

Results are saved to a pandas DataFrame for analysis.



## Notes

* Using float16 for the CLIP model reduces the GPU memory usage
* Tested on T4 GPU (colab free tier): embedding speed 80 Images/sec with a batch size of 256 
* Could experiment with different faiss index configurations
* k=50 gives the top 50 nearest matches for each query image


## Next steps
 - [ ] Try other Image embedding models like BLIP 2 etc.
 - [ ] Train a model on ground up with negative samples on CIFAR 10 to compare
   - [x] ViT backbone vs ResNet

