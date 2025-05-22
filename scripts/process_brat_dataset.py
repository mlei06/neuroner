import pandas as pd
import os
import shutil
import random
import argparse

def split_records(input_file):
    """Split the input file by ||||END_OF_RECORD and return list of (record_id, text)"""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    records = content.split('||||END_OF_RECORD')
    result = []

    for record in records:
        try:
            i = record.index("START_OF_RECORD") + len("START_OF_RECORD") + 1
        except ValueError:
            continue  

        # get record id
        temp = i
        count = 0
        while count < 8 and temp < len(record):
            if record[temp] == "|":
                count += 1
            temp += 1

        record_id = record[i:temp]
        text = record[temp:].strip().replace("\n", " ")
        result.append((record_id, text))

    return result

def convert_to_brat(text_path, annotation_csv, output_dir):
    """Convert raw text and CSV annotations to BRAT format"""
    print(f"Reading from: {text_path}")
    records = split_records(text_path)
    df = pd.read_csv(annotation_csv)
    os.makedirs(output_dir, exist_ok=True)

    for i, (record_id, text) in enumerate(records):
        entities = []
        matched_rows = df[df['record_id'] == record_id]

        for _, row in matched_rows.iterrows():
            start = row['begin']
            end = start + row['length']
            entity = row['type']

            if end > len(text):
                continue  # Skip annotation if it's out of bounds

            span_text = text[start:end]
            entities.append((entity, start, end, span_text))

        
        txt_path = os.path.join(output_dir, f"note_{i+1}.txt")
        ann_path = os.path.join(output_dir, f"note_{i+1}.ann")

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        with open(ann_path, 'w', encoding='utf-8') as f:
            for j, (label, start, end, span_text) in enumerate(entities):
                f.write(f"T{j}\t{label} {start} {end}\t{span_text}\n")

def modify_labels(input_dir, output_dir):
    """Update BRAT annotations to use  label mappings"""
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    label_mappings = {
        'NAME': 'NAME',
        'LOCATION': 'LOCATION',
        'CITY': 'CITY',
        'STATE': 'STATE',
        'ID': 'ID',
        'ORGANIZATION': 'ORGANIZATION',
        'PROFESSION': 'PROFESSION',
        'DATE': 'DATE',
        'AGE': 'AGE',
        'COUNTRY': 'COUNTRY',
        'HOSPITAL': 'HOSPITAL',
        'PHONE': 'PHONE',
        
    }

    for split in ['train', 'val', 'test']:
        input_split_dir = os.path.join(input_dir, split)
        output_split_dir = os.path.join(output_dir, split)

        print(f"\nProcessing {split} set...")

        # Gather all file prefixes
        record_ids = set(os.path.splitext(f)[0] for f in os.listdir(input_split_dir) if f.endswith(".ann"))

        for record_id in record_ids:
            # Copy .txt file unchanged
            txt_src = os.path.join(input_split_dir, f"{record_id}.txt")
            txt_dst = os.path.join(output_split_dir, f"{record_id}.txt")
            shutil.copy2(txt_src, txt_dst)

            # Modify .ann file
            ann_src = os.path.join(input_split_dir, f"{record_id}.ann")
            ann_dst = os.path.join(output_split_dir, f"{record_id}.ann")

            with open(ann_src, 'r', encoding='utf-8') as src, \
                 open(ann_dst, 'w', encoding='utf-8') as dst:
                for line in src:
                    if line.startswith('T'):
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            ann_details = parts[1].split()
                            if len(ann_details) >= 2:
                                entity_type = ann_details[0]
                                if entity_type in label_mappings:
                                    ann_details[0] = label_mappings[entity_type]
                                    parts[1] = ' '.join(ann_details)
                                    line = '\t'.join(parts) + '\n'
                    dst.write(line)

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split BRAT dataset into train/val/test sets"""
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)

    # Create subdirs
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    ann_files = [f for f in os.listdir(input_dir) if f.endswith(".ann")]
    random.shuffle(ann_files)

    total_files = len(ann_files)
    train_cutoff = int(total_files * train_ratio)
    val_cutoff = train_cutoff + int(total_files * val_ratio)

    train_files = ann_files[:train_cutoff]
    val_files = ann_files[train_cutoff:val_cutoff]
    test_files = ann_files[val_cutoff:]

    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # Copy files to their respective directories
    for files, dst_dir in [(train_files, train_dir), (val_files, val_dir), (test_files, test_dir)]:
        for ann_file in files:
            txt_file = ann_file.replace(".ann", ".txt")
            if txt_file in os.listdir(input_dir):
                shutil.copy(os.path.join(input_dir, ann_file), os.path.join(dst_dir, ann_file))
                shutil.copy(os.path.join(input_dir, txt_file), os.path.join(dst_dir, txt_file))

def process_dataset(text_path, annotation_csv, output_dir, apply_mappings=True, temp_dir="temp_brat"):
    """Process the entire dataset: convert to BRAT, modify labels, and split"""
    #Convert to BRAT format
    
    convert_to_brat(text_path, annotation_csv, temp_dir)
    
    # Modify labels (if requested)
    if apply_mappings:
        
        modified_dir = os.path.join(temp_dir, "modified")
        modify_labels(temp_dir, modified_dir)
        split_input_dir = modified_dir
    else:
        
        split_input_dir = temp_dir
    
    #Split dataset
   
    split_dataset(split_input_dir, output_dir)
    
   
    shutil.rmtree(temp_dir)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset: convert to BRAT, modify labels, and split")
    parser.add_argument("--text", required=True, help="Path to input text file (e.g., orig.txt)")
    parser.add_argument("--annotations", required=True, help="Path to CSV file with annotations (e.g., train.csv)")
    parser.add_argument("--output", required=True, help="Directory to save final processed dataset")
    parser.add_argument("--mappings", action="store_true", help="Apply label mappings (default: False)")
    
    args = parser.parse_args()
    process_dataset(args.text, args.annotations, args.output, args.mappings) 