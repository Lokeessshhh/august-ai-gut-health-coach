import json
from openai import OpenAI
import time

# Initialize NVIDIA API client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

def call_llm(prompt):
    """Call the NVIDIA LLM API and return the response"""
    try:
        completion = client.chat.completions.create(
            model="meta/llama3-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=True
        )
        
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        
        return response.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

def generate_negative_example(instruction):
    """Generate a negative and positive response pair"""
    prompt = f"""Given this health/nutrition instruction: "{instruction}"

Create a JSON response with exactly this format:
{{"bad_response": "unsafe, incomplete, or dismissive answer", "good_response": "safe, supportive, and informative answer"}}

Rules:
- bad_response: Should be unsafe, dismissive, incomplete, or potentially harmful
- good_response: Should be safe, supportive, informative, and remind to see a doctor for severe symptoms
- Return ONLY the JSON object, nothing else"""

    response = call_llm(prompt)
    if response:
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
    return None

def generate_tone_example(instruction):
    """Generate an empathetic, friendly, supportive response"""
    prompt = f"""Given this health/nutrition instruction: "{instruction}"

Write a response that is:
- Empathetic and friendly
- Supportive and understanding
- Factually correct and clear
- Safe (reminds to see doctor if symptoms are severe)
- Warm and encouraging

Return ONLY the response text, no extra formatting or explanations."""

    response = call_llm(prompt)
    return response if response else None

def process_dataset():
    """Process the dataset and generate both JSONL files"""
    
    # Read the dataset
    try:
        with open('data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: data.json file not found")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON in data.json")
        return
    
    # Handle both single object and list of objects
    if isinstance(data, dict):
        instructions = [data['instruction']]
    else:
        instructions = [item['instruction'] for item in data if 'instruction' in item]
    
    print(f"Processing {len(instructions)} instructions...")
    
    # Open both files for writing (append mode)
    with open('negatives.jsonl', 'w', encoding='utf-8') as neg_file, \
         open('tone_examples.jsonl', 'w', encoding='utf-8') as tone_file:
        
        for i, instruction in enumerate(instructions):
            print(f"\n--- Processing instruction {i+1}/{len(instructions)} ---")
            print(f"Instruction: {instruction}")
            
            # Step 1: Generate negative example for this instruction
            print("Generating negative example...")
            negative_result = generate_negative_example(instruction)
            if negative_result:
                negatives_entry = {
                    "instruction": instruction,
                    "bad_response": negative_result.get("bad_response", ""),
                    "good_response": negative_result.get("good_response", "")
                }
                # Write immediately to negatives.jsonl
                neg_file.write(json.dumps(negatives_entry) + '\n')
                neg_file.flush()  # Ensure it's written to disk
                print("✓ Negative example written to negatives.jsonl")
            else:
                print("✗ Failed to generate negative example")
            
            # Small delay between calls
            time.sleep(0.5)
            
            # Step 2: Generate tone example for this instruction
            print("Generating tone example...")
            tone_response = generate_tone_example(instruction)
            if tone_response:
                tone_entry = {
                    "instruction": instruction,
                    "response": tone_response
                }
                # Write immediately to tone_examples.jsonl
                tone_file.write(json.dumps(tone_entry) + '\n')
                tone_file.flush()  # Ensure it's written to disk
                print("✓ Tone example written to tone_examples.jsonl")
            else:
                print("✗ Failed to generate tone example")
            
            # Delay before next instruction
            print(f"Completed instruction {i+1}/{len(instructions)}")
            if i < len(instructions) - 1:  # Don't delay after the last instruction
                time.sleep(1)
    
    print(f"\n🎉 All instructions processed!")
    print(f"Check the generated files:")
    print(f"- negatives.jsonl")
    print(f"- tone_examples.jsonl")

if __name__ == "__main__":
    process_dataset()