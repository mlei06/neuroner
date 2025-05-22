import argparse

def get_false_negatives(file, label):
    with open(file, "r") as f:
        text = f.read()
    true_positives = []
    false_negatives = []
    false_positives = []
    falsenegativescontext = []
    noteset = set()
    fixedtext = []
    for line in text.split("\n"):
        if line:
            text = line.split(" ")[0]
            note = line.split(" ")[1]
            gold = line.split(" ")[4]
            pred = line.split(" ")[5]
            if note not in noteset:
                noteset.add(note)
                prev_text = None
            if label == "PATIENT":
                if "PATIENT" in gold and ("PATIENT" in pred or "DOCTOR" in pred):
                    true_positives.append(text)
                elif "PATIENT" in gold and ("PATIENT" not in pred and "DOCTOR" not in pred):
                    false_negatives.append(text)
                elif "PATIENT" not in gold and ("PATIENT" in pred or "DOCTOR" in pred):
                    false_positives.append(text)
            else:
                if label in gold and label in pred:
                    true_positives.append(text)
                elif label in gold and label not in pred:
                    false_negatives.append(text)
                elif label not in gold and label in pred:
                    false_positives.append(text)
    return false_negatives, true_positives, false_positives

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update BRAT annotations with label mappings")
    parser.add_argument("--file", required=True, help="Input file")
    parser.add_argument("--label", required=True, help="Label to extract")
    
    args = parser.parse_args()
    file = args.file
    label = args.label
    false_negatives, true_positives, false_positives = get_false_negatives(file, label)
    print(f"False Negatives: {len(false_negatives)}")
    print(false_negatives)
    print("--------------------------------")
    print(f"True Positives: {len(true_positives)}")
    print(true_positives)
    print("--------------------------------")
    print(f"False Positives: {len(false_positives)}")
    print(false_positives)
    print("--------------------------------")
    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    recall = len(true_positives) / (len(false_negatives) + len(true_positives))
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")









