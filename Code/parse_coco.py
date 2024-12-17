import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(clip_model_type: str):
    #device = torch.device('cuda:0')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"C:/Users/admitos/Desktop/ThesisUU/pretrained_models/test.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    path_to_dii_dii = 'C:/Users/admitos/Desktop/ThesisUU/DII-SIS/dii_sis_test_annots.json'
    with open(path_to_dii_dii, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i][0]
        photo_id = d["photo_flickr_id"]
        if photo_id[0]!=str(7) and photo_id[1]!=str(6):
            print("here we skip file with photo_id: ", photo_id)
            continue
        else:
            #filename = f"C:/Users/admitos/Desktop/ThesisUU/vist_yingjin_images/{int(photo_id)}.jpg"
            filename = f"F:/ThesisUU/test/{int(photo_id)}.jpg"
            if not os.path.isfile(filename):
                print("\n", "No File Found")
                continue
                filename = f"C:/Users/admitos/Desktop/ThesisUU/vist_yingjin_images/{int(photo_id)}.jpg"
            else:
                print("\n We found a file: ", photo_id)
                image = io.imread(filename)
                image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
                with torch.no_grad():
                    prefix = clip_model.encode_image(image).cpu()
                d["clip_embedding"] = i
                all_embeddings.append(prefix)
                all_captions.append(d)
                if (i+1)%1000 == 0:
                    with open(out_path, 'wb') as f:
                        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type)) 
