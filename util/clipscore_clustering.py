import torch
from PIL import Image
import open_clip
import numpy as np
import os
from tqdm import tqdm

def spectral_topk(S, k):
    vals, vecs = np.linalg.eigh(S)
    top_vec = vecs[:, np.argmax(vals)]
    return np.argsort(top_vec)[-k:]

def clipscore_clustering(candidate_images, top_k=10, image_dir=None):
    """
    Given a small set of candidate images (within top-activated ones) for each prototype,
    find a subset of images that are similar to each other based on CLIP score,
    and use them as the final cluster for the prototype.
    """

    filtered_images = []
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to('cuda')
    model.eval()

    for proto_id in tqdm(range(candidate_images.shape[1]), desc="Spectral clustering for` prototypes"):
        cur_images = candidate_images[:, proto_id] # (num_images,)
        # compute pairwise CLIP scores between images
        images = []
        for img_name in cur_images:
            img = preprocess(Image.open(os.path.join(image_dir, img_name + '.jpg')))
            images.append(img)

        images = torch.stack(images).to('cuda')

        with torch.no_grad(), torch.autocast("cuda"):
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)

        # Compute full similarity matrix in ONE matmul
        clip_scores = (features @ features.T).cpu().numpy()

        # now run the spectral clustering algorithm on the clip_scores to find clusters of similar images
        topk_idx = spectral_topk(clip_scores, top_k)
        filtered_images.append(cur_images[topk_idx])
    
    filtered_images = np.stack(filtered_images, axis=1) # (num_images, num_prototypes)
    return filtered_images
