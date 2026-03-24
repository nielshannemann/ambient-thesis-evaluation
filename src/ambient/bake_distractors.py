import json
from pathlib import Path
# Assuming your create_distractor function is in a file called distractors.py
from external.ambient.evaluation.distractors import create_distractor

def bake_distractors(input_path="external/ambient/AmbiEnt/test.jsonl", output_path="external/ambient/AmbiEnt/test_baked.jsonl"):
    print(f"Baking distractors from {input_path}...")
    baked_data = []
    
    # Set a strict seed just for this script so the generation is reproducible
    import random
    random.seed(42) 
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            
            # Check the boolean flags and generate distractors for the correct targets
            if row.get('premise_ambiguous'):
                row['distractor_premise'] = create_distractor(row['premise'])
                
            if row.get('hypothesis_ambiguous'):
                row['distractor_hypothesis'] = create_distractor(row['hypothesis'])
            
            baked_data.append(row)
            
    # Write the new dataset to disk
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in baked_data:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            
    print(f"Successfully baked {len(baked_data)} distractors into {output_path}!")

if __name__ == "__main__":
    bake_distractors()