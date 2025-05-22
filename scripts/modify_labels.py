# modify_labels.py

import os
import shutil
import argparse

def modify_labels(input_dir, output_dir):
    """Update BRAT annotations to use label mappings"""
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    label_mappings = {
        'NAME': 'PATIENT',
        'LOCATION': 'LOCATION_OTHER',
        'CITY': 'LOCATION_OTHER',
        'STATE': 'LOCATION_OTHER',
        'ID': 'IDNUM',
        'ORGANIZATION': 'HOSPITAL',
        'PROFESSION': 'DOCTOR',
        'DATE': 'DATE',
        'AGE': 'AGE',
        'COUNTRY': 'LOCATION_OTHER',
        'HOSPITAL': 'HOSPITAL',
        'PHONE': 'PHONE'
    }

    for split in ['train', 'val', 'test']:
        input_split_dir = os.path.join(input_dir, split)
        output_split_dir = os.path.join(output_dir, split)

        if not os.path.exists(input_split_dir):
            print(f"Warning: {split} directory not found in {input_dir}")
            continue

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update BRAT annotations with label mappings")
    parser.add_argument("--input", required=True, help="Input directory containing train/val/test folders")
    parser.add_argument("--output", required=True, help="Output directory for modified annotations")
    
    args = parser.parse_args()
    modify_labels(args.input, args.output)

    # Final report
    print("\nLabel modifications completed!")
    print(f"Original dataset: {args.input}")
    print(f"Modified dataset: {args.output}")
    
