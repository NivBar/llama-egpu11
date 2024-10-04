import pickle

# Function to merge two dictionaries
def merge_dicts(dict1, dict2):
    for primary_key, secondary_dict in dict2.items():
        if primary_key not in dict1:
            dict1[primary_key] = secondary_dict
        else:
            # Merge the secondary level
            for secondary_key, value in secondary_dict.items():
                if secondary_key not in dict1[primary_key]:
                    dict1[primary_key][secondary_key] = value
    return dict1

# Load the dictionaries from pkl files
def load_dict_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Save the combined dictionary back to a pkl file
def save_dict_to_pkl(dict_data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dict_data, file)

# File paths for your pkl files
file1 = 'cache_folder/dict11.pkl'
file2 = 'cache_folder/dict12.pkl'
output_file = 'cache_folder/cache_offline.pkl'

# Load the two dictionaries
dict1 = load_dict_from_pkl(file1)
dict2 = load_dict_from_pkl(file2)

# Merge the dictionaries
combined_dict = merge_dicts(dict1, dict2)

# Save the combined dictionary to a new pkl file
save_dict_to_pkl(combined_dict, output_file)

print("Dictionaries combined and saved to", output_file)
