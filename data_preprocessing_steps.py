import os
import fnmatch
import pandas as pd
import json

#########################################
### Data Preprocessing
#######################################
def delete_non_filtered_tif_files(folder_path):
    # Deletes all files starting with 'Non-Filtered' and with the '.tif' extension
    # from the specified folder and its subfolders.
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # Traverse the folder and its subfolders
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # Check if the file matches the naming criteria
            if fnmatch.fnmatch(filename, 'Non-Filtered*.tif'):
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting file '{file_path}': {e}")

if __name__ == "__main__":
    
    target_folder = './' 
    delete_non_filtered_tif_files(target_folder)

path_list = []
for root, dirs, files in os.walk('./'):
    # print(root)
    if len(root)>2:
        path_list.append(root)

print(path_list)

for path in path_list:
    # path = './Random_series_001'
    data = pd.read_csv(f'{path}/image_data.csv')
    df = pd.DataFrame(data)
    df = df.drop(['BaS', 'Sn2BaS3', 'SnS', 'BaS_thickness_cm', 'Sn2BaS3_thickness_cm', 'SnS_thickness_cm'], axis=1)

    with open(f'{path}/settings.json', 'r') as f:
        settings = json.load(f)

    boundary1 = settings['boundaries'][1]
    boundary2 = settings['boundaries'][2]

    if 'boundaries' in settings and len(settings['boundaries']) > 2:
            boundary = pd.DataFrame({
            'boundary1': [boundary1],
            'boundary2': [boundary2]
        })
    else:
        raise ValueError("Settings do not contain valid boundaries.")
    df['target'] = df['F'].apply(lambda x: 1 if x < boundary1 else 2 if boundary1 <= x <= boundary2 else 3)

    df.to_csv(f'{path}/image_data_1.csv', index=False)

print(df.head())
print(settings['boundaries'])


###########################################
## TRAIN - VAL - TEST
###########################################

# path_list = []
# for root, dirs, files in os.walk('./'):
#     # print(root)
#     if len(root)>2:
#         path_list.append(root)


# train_directories = path_list[:100]
# val_directories = path_list[100:150]
# test_directories = path_list[150:]

# print(train_directories)
# print(val_directories)
# print(test_directories)

# combined_data = pd.DataFrame()
# for directory in train_directories:
#     file_path = os.path.join(directory, 'image_data_1.csv')
    
#     if os.path.exists(file_path):
#         df = pd.read_csv(file_path)
        
#         combined_data = pd.concat([combined_data, df], ignore_index=True)
#     else:
#         print(f"File not found: {file_path}")

# combined_data.to_csv('train_data.csv', index=False)

# print("Data combined successfully into 'train_data.csv'.")

# combined_data = pd.DataFrame()
# for directory in val_directories:
#     file_path = os.path.join(directory, 'image_data_1.csv')
    
#     if os.path.exists(file_path):
#         df = pd.read_csv(file_path)
        
#         combined_data = pd.concat([combined_data, df], ignore_index=True)
#     else:
#         print(f"File not found: {file_path}")

# combined_data.to_csv('val_data.csv', index=False)

# print("Data combined successfully into 'val_data.csv'.")

# combined_data = pd.DataFrame()
# for directory in test_directories:
#     file_path = os.path.join(directory, 'image_data_1.csv')
    
#     if os.path.exists(file_path):
#         df = pd.read_csv(file_path)
        
#         combined_data = pd.concat([combined_data, df], ignore_index=True)
#     else:
#         print(f"File not found: {file_path}")

# combined_data.to_csv('test_data.csv', index=False)

# print("Data combined successfully into 'test_data.csv'.")