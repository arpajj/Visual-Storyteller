
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import json

class Scorer():
    def __init__(self,ref,gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),

        ]
    
    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f"%(m, sc))
                total_scores["Bleu"] = score
            else:
                print("%s: %0.3f"%(method, score))
                total_scores[method] = score
        
        print('*****DONE*****')
        for key,value in total_scores.items():
            print('{}:{}'.format(key,value))

if __name__ == '__main__':
    pred_fn = 'C:/Users/admitos/Desktop/ThesisUU/Results/zero-shot/Transformer/generated_dict_no_beam.json'
    gt_fn = 'C:/Users/admitos/Desktop/ThesisUU/Results/zero-shot/Transformer/original_dict.json'

    with open(gt_fn, 'r') as file:
        gts = json.load(file)
    with open(pred_fn, 'r') as file:
        res = json.load(file) 

    gens = {}
    for img_id, caption in res.items():
        gens[img_id] = [caption]
    
    scorer = Scorer(gens,gts)
    scorer.compute_scores()




