from pycocoevalcap.spice.spice import Spice
spice_evaluator = Spice()

def evaluate_captions(ref_captions, pred_caption):
    # Format the captions as expected by the SPICE evaluator
    hypo = {0: [pred_caption]}
    ref = {0: ref_captions}

    # Evaluate SPICE score
    score, scores = spice_evaluator.compute_score(ref, hypo)
    return score

# Example usage
reference_captions = [
    "A group of people standing around a kitchen preparing food.",
    "A family is cooking dinner in a kitchen filled with cooking utensils.",
    "Several people are at a stove preparing a meal."
]
predicted_caption = "A group of people are cooking in a kitchen."

spice_score = evaluate_captions(reference_captions, predicted_caption)
print(f"SPICE Score: {spice_score}")
