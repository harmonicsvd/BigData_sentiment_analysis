import json

# Step 1: Read the input JSON file
input_file = '/Users/varadkulkarni/TUHH/SecondSemester/BigData/Project_data/filtered_comments.json'
with open(input_file, 'r') as file:
    data = json.load(file)

# Step 2: Generate 50,000 entries
required_entries = 50000
current_length = len(data)
new_data = (data * (required_entries // current_length)) + data[:required_entries % current_length]

# Step 3: Write to a new JSON file
output_file = '/learning/trainedmodel/datasource.json'
with open(output_file, 'w') as file:
    json.dump(new_data, file, indent=4)
