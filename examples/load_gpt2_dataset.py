import json

def load_gpt2_dataset(json_file_name, num_examples=float('inf')):
    texts = []
    for i, line in enumerate(open(json_file_name)):
        if i >= num_examples:
            break
        texts.append(json.loads(line)['text'])
    return texts