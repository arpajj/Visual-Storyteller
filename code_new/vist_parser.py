import pickle
import json
import torch
import os
import skimage.io as io
from tqdm import tqdm
import argparse
import clip
from PIL import Image

def main(clip_model_type):
    #device = torch.device('cuda:0')
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"path to your .pkl file where you would like to store the CLIP embeddings of for Train, Validation, Test sets of VIST. See example below"
    # out_path = f"/data/admitosstorage/CLIP_image_embeddings/val_vit.pkl" 
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    path_to_dii f"path to where you have stored the DII annotatations from VIST dataset. Respectively for Train, Val, Test. See the example below" 
    #path_to_dii = '/data/admitosstorage/DII-annotation/val.description-in-isolation.json'
    with open(path_to_dii, 'r', encoding='utf-8') as f:
        data = json.load(f)['annotations']
        
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    q = 0
    for i in tqdm(range(len(data))):
        dictionary = data[i][0]
        photo_id = dictionary["photo_flickr_id"]
        filename = f"path to where you have stored the actual images of VIST for each set (Train, Val, Test) respectively. See example below"
        #filename = f"/data/admitosstorage/val_images/{int(photo_id)}.jpg"
        if not os.path.isfile(filename):
            print("No File Found")
            continue
        else:
            print("We found a file on", i, ": ", photo_id)
            image = io.imread(filename)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).to(device)
            dictionary["clip_embedding"] = q
            q += 1
            all_embeddings.append(prefix)
            all_captions.append(dictionary)
            if (i+1)%1000 == 0:
                with open(out_path, 'wb') as f:
                    pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
        

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    # print(f"Choose the type of the CLIP encoder from the following choices: ['RN50', 'RN101', 'RN50x4', 'ViT-B/32']")
    # x = input()
    # if x not in  ['RN50', 'RN101', 'RN50x4', 'ViT-B/32']:
    #     x = 'ViT-B/32'
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    print("The type of CLIP encoder is: ", args.clip_model_type)
    exit(main(args.clip_model_type))
    #exit(main(x))


