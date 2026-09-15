"""Add reproducible AMBIENT distractors using the benchmark implementation."""

import json
from pathlib import Path

from external.ambient.evaluation.distractors import create_distractor


def bake_distractors(
    input_path="external/ambient/AmbiEnt/test.jsonl",
    output_path="external/ambient/AmbiEnt/test_baked.jsonl",
):
    print(f"Baking distractors from {input_path}...")
    baked_data = []
    
    # Match the fixed seed used to prepare the tracked dataset.
    import random
    random.seed(42) 
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            
            # Add a distractor only for each sentence marked as ambiguous.
            if row.get('premise_ambiguous'):
                row['distractor_premise'] = create_distractor(row['premise'])
                
            if row.get('hypothesis_ambiguous'):
                row['distractor_hypothesis'] = create_distractor(row['hypothesis'])
            
            baked_data.append(row)
            
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in baked_data:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            
    print(f"Successfully baked {len(baked_data)} distractors into {output_path}!")


def run(args) -> int:
    bake_distractors(input_path=args.data_path, output_path=args.output_path)
    return 0
