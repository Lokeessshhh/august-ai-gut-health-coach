import json

def json_to_jsonl(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # Load the JSON file

    with open(output_file, "w", encoding="utf-8") as f:
        if isinstance(data, list):
            # If JSON is an array of objects
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            # If JSON is a single object
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

# Example usage
json_to_jsonl("data.json", "data.jsonl")
