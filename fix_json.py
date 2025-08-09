import json

def fix_json_file(input_path, output_path):
    # Read the original incorrect JSON
    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Insert commas between objects and wrap in array []
    corrected = '[' + ','.join(line.strip() for line in lines if line.strip()) + ']'

    # Validate the JSON before saving
    try:
        # Try to parse the JSON to ensure it's valid
        json.loads(corrected)
        
        # Save corrected file
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(corrected)
        print(f"Fixed file saved as {output_path}")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ Error: Could not create valid JSON: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "Entity Recognition in Resumes.json"
    output_file = "Entity_Recognition_Fixed.json"
    fix_json_file(input_file, output_file)
