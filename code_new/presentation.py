from transformers import BartForConditionalGeneration, BartTokenizer
import torch, os
import helper as help

# LOCALLY
#WEIGHTS_PATH_CC = "C:/Users/yugio/OneDrive/Desktop/ThesisUU/pretrained_models/MSCOCO/Transformer/dii_trO_rn-003.pt" # path from pre-trained in COCO 
WEIGHTS_PATH_CC = "C:/Users/yugio/OneDrive/Desktop/ThesisUU/pretrained_models/MSCOCO/Transformer/dii_trG_rn-003.pt" # path from pre-trained in COCO 
WEIGHTS_PATH_BART = 'C:/Users/yugio/OneDrive/Desktop/ThesisUU/Phase_2/trained_models/BART/trained_bart_e9.pt' # path from pre-trained in COCO 

print("LOOOOOOOOOOOOOL XDDDDD")
print("Device used:", help.D.upper())

# Load Vision-to-Caption model
predictor = help.Predictor()
my_clipcap_model = predictor.setup(WEIGHTS_PATH_CC)
print("1) Vision-to-Caption Model Loaded Succesfully!")

# Load Caption-to-Story model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model.load_state_dict(torch.load(WEIGHTS_PATH_BART, map_location=help.D, weights_only=True))
model.eval()
print("2) Caption-to-Story Model Loaded Succesfully!")

print("Here goes my story...")
my_story_path = 'C:/Users/yugio/OneDrive/Desktop/ThesisUU/My_Visual_Story/'
#my_story_path = 'C:/Users/yugio/OneDrive/Desktop/ThesisUU/Comparisons/'
#my_story_path = 'C:/Users/yugio/OneDrive/Desktop/ThesisUU/Visual Stories/Visual Story 6/'

story_captions = []
for i, filename in enumerate(os.listdir(my_story_path)): 
    if filename.endswith(".jpg") or filename.endswith(".png"):
        final_img_path = os.path.join(my_story_path, filename)
        print(f"Prediction on image {i}")
        indiv_cap = predictor.predict(final_img_path, my_clipcap_model, help.USE_BEAM_SEARCH)
        story_captions.append(indiv_cap) 

print("Almost there...")
input_text = ' </s> '.join(story_captions)
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
with torch.no_grad():
    #summary_ids = model.generate(input_ids.to(help.D), max_length=200, early_stopping=False, do_sample=False) ### Greedy search
    #summary_ids = model.generate(input_ids.to(help.D), max_length=200, num_beams=1, early_stopping=False, do_sample=True) ### multinomial sampling
    summary_ids = model.generate(input_ids.to(help.D), max_length=200, num_beams=1, early_stopping=False, do_sample=True, top_p=0.9) ### nucleus sampling
    #summary_ids = model.generate(input_ids.to(help.D), max_length=200, num_beams=5, early_stopping=True) ### beam search
    story = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print()
print(story)