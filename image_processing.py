import os
import pandas as pd
import pickle
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np


def enhance_contrast_and_brightness(img, contrast_factor=2.0, brightness_factor=2.5):
    brightness_enhancer = ImageEnhance.Brightness(img)
    brightened_img = brightness_enhancer.enhance(brightness_factor)

    contrast_enhancer = ImageEnhance.Contrast(brightened_img)
    enhanced_img = contrast_enhancer.enhance(contrast_factor)

    return enhanced_img

def combine_data_and_images(directories, data_type):
    # Parameters to crop image
    left = 135
    upper = 13
    right = 740
    lower = 618
    crop_box = (left, upper, right, lower)

    combined_data = pd.DataFrame()
    df_list = []
    images_T = []
    images_R = []
    enhanced_image_T = []
    enhanced_image_R = []
    boundary_list = []

    for directory in directories:
        # Construct paths for CSV and image files
        file_path = os.path.join(directory, 'image_data_1.csv')
        image_path_T = os.path.join(directory, 'Filtered_T_Image.tif')
        image_path_R = os.path.join(directory, 'Filtered_R_Image.tif')
        boundary_path = os.path.join(directory, 'boundary.csv')

        # Check if the CSV file exists
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df_list.append(df)
            combined_data = pd.concat([combined_data, df], ignore_index=True)
        else:
            print(f"File not found: {file_path}")

        # Check if the image file exists
        if os.path.exists(image_path_T):
            img_T = Image.open(image_path_T).crop(crop_box)
            enhanced_img_T = enhance_contrast_and_brightness(img_T)
            images_T.append(img_T)
            enhanced_image_T.append(enhanced_img_T)
        else:
            print(f"Image not found: {image_path_T}")

        if os.path.exists(image_path_R):
            img_R = Image.open(image_path_R).crop(crop_box)
            enhanced_img_R = enhance_contrast_and_brightness(img_R)
            images_R.append(img_R)
            enhanced_image_R.append(enhanced_img_R)
        else:
            print(f"Image not found: {image_path_R}")

        # Check if the boundary CSV file exists
        if os.path.exists(boundary_path):
            boundary = pd.read_csv(boundary_path)
            boundary_list.append(boundary.values.flatten())
        else:
            print(f"File not found: {boundary_path}")

    # Save the combined DataFrame to a CSV file
    output_csv = f'{data_type}_data.csv'
    combined_data.to_csv(output_csv, index=False)
    print(f"Data combined successfully into '{output_csv}'.")

    # Save lists to files
    with open(f'{data_type}_images_T_list.pkl', 'wb') as f:
        pickle.dump(images_T, f)
    with open(f'{data_type}_images_R_list.pkl', 'wb') as f:
        pickle.dump(images_R, f)
    with open(f'{data_type}_enhanced_images_T_list.pkl', 'wb') as f:
        pickle.dump(enhanced_image_T, f)
    with open(f'{data_type}_enhanced_images_R_list.pkl', 'wb') as f:
        pickle.dump(enhanced_image_R, f)
    with open(f'{data_type}_df_list.pkl', 'wb') as f:
        pickle.dump(df_list, f)
    with open(f'{data_type}_boundary_list.pkl', 'wb') as f:
        pickle.dump(boundary_list, f)

    return images_T, images_R, enhanced_image_T, enhanced_image_R, df_list, boundary_list

def load_data_and_images(data_type):
    with open(f'{data_type}_images_T_list.pkl', 'rb') as f:
        images_T = pickle.load(f)
    with open(f'{data_type}_images_R_list.pkl', 'rb') as f:
        images_R = pickle.load(f)
    with open(f'{data_type}_enhanced_images_T_list.pkl', 'rb') as f:
        enhanced_image_T = pickle.load(f)
    with open(f'{data_type}_enhanced_images_R_list.pkl', 'rb') as f:
        enhanced_image_R = pickle.load(f)
    with open(f'{data_type}_df_list.pkl', 'rb') as f:
        df_list = pickle.load(f)
    with open(f'{data_type}_boundary_list.pkl', 'rb') as f:
        boundary_list = pickle.load(f)
    
    return images_T, images_R, enhanced_image_T, enhanced_image_R, df_list, boundary_list

def plot_targets_on_simulated_image(df):
    # Determine the range of coordinates
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    # print(x_min, x_max, y_min, y_max)
    
    # Create a blank white image based on the coordinate range
    img_width = int(x_max - x_min)
    img_height = int(y_max - y_min)
    img = Image.new('RGB', (img_width, img_height), 'white')
    # Create a figure and axis
    _, ax = plt.subplots()

    # Display the blank image
    ax.imshow(img, extent=[x_min, x_max, y_min, y_max], aspect='equal')

    # Plot points based on target values
    colors = {1: 'lightblue', 2: 'plum', 3: 'tomato'}
    
    # Plot each unique coordinate for each target
    for target in colors.keys():
        target_points = df[df['target'] == target]
        
        # Use the coordinates to plot the points on the image
        ax.scatter(target_points['x'], target_points['y'], 
                   color=colors[target], 
                   label=f'Target {target}', 
                   s=10,
                   alpha=0.5)

    # Set the axis limits to match the coordinate range
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    # Encode the target labels
    le = LabelEncoder()
    df['target_encoded'] = le.fit_transform(df['target'])

    # Fit logistic regression model
    X = df[['x', 'y']]
    y = df['target_encoded']
    model = LogisticRegression(multi_class='multinomial')
    model.fit(X, y)
    
    # Create a mesh to plot the decision boundaries
    xx, yy = np.meshgrid(np.linspace(x_min - 1, x_max + 1, 100), np.linspace(y_min - 1, y_max + 1, 100))
    
    # Ensure xx and yy are DataFrames with the appropriate column names
    grid_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['x', 'y'])
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot decision boundaries
    ax.contour(xx, yy, Z, levels=[0.5, 1.5], colors=['darkgreen', 'darkgreen'], linestyles=['-', '-'])

    # Add labels and legend
    ax.set_title('"True" Decision Boundaries')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Show the plot
    plt.show()


def plot_targets_on_real_image(df, img):

    # Create a figure and axis
    _, ax = plt.subplots()

    # Display the image
    ax.imshow(img, extent=[df['x'].min(), df['x'].max(), df['y'].min(), df['y'].max()])
    


    # Plot points based on target values
    colors = {1: 'lightblue', 2: 'plum', 3: 'tomato'}
    
    # Plot each unique coordinate for each target
    for target in colors.keys():
        target_points = df[df['target'] == target]
        
        # Use the coordinates to plot the points on the image
        ax.scatter(target_points['x'], target_points['y'], 
                   color=colors[target], 
                   label=f'Target {target}', 
                   s=10,
                   alpha = 0)

    # Set the axis limits to match the coordinate range
    ax.set_xlim([df['x'].min(), df['x'].max()])
    ax.set_ylim([df['y'].min(), df['y'].max()])

    # Encode the target labels
    le = LabelEncoder()
    df['target_encoded'] = le.fit_transform(df['target'])

    # Fit logistic regression model
    X = df[['x', 'y']]
    y = df['target_encoded']
    model = LogisticRegression(multi_class='multinomial')
    model.fit(X, y)
    
    # Create a mesh to plot the decision boundaries
    x_min, x_max = df['x'].min() - 1, df['x'].max() + 1
    y_min, y_max = df['y'].min() - 1, df['y'].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    # Ensure xx and yy are DataFrames with the appropriate column names
    grid_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['x', 'y'])
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot decision boundaries
    ax.contour(xx, yy, Z, levels=[0.5, 1.5], colors=['darkgreen', 'darkgreen'], linestyles=['-', '-'])

    # Add labels and legend
    ax.set_title('True Decision Boundaries')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Show the plot
    plt.show()



def plot_combined_targets_on_image(df, img):
    # img_width, img_height = img.size

    # Create a figure and axis
    _, ax = plt.subplots()

    # Display the image
    ax.imshow(img, extent=[df['x'].min(), df['x'].max(), df['y'].min(), df['y'].max()])

    # Plot points based on target values
    colors = {1: 'lightblue', 2: 'plum', 3: 'tomato'}
    
    # Plot each unique coordinate for each target
    for target in colors.keys():
        target_points = df[df['target'] == target]
        ax.scatter(target_points['x'], target_points['y'], 
                   color=colors[target], 
                   label=f'True Target {target}', 
                   s=10,
                   alpha = 0.5)

    # Set the axis limits to match the coordinate range
    ax.set_xlim([df['x'].min(), df['x'].max()])
    ax.set_ylim([df['y'].min(), df['y'].max()])

    # Encode the true target labels
    le_true = LabelEncoder()
    df['target_encoded_true'] = le_true.fit_transform(df['target'])

    # Fit logistic regression model for true targets
    X = df[['x', 'y']]
    y_true = df['target_encoded_true']
    model_true = LogisticRegression()
    model_true.fit(X, y_true)

    # Create a mesh to plot the true decision boundaries
    x_min, x_max = df['x'].min() - 1, df['x'].max() + 1
    y_min, y_max = df['y'].min() - 1, df['y'].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['x', 'y'])
    Z_true = model_true.predict(grid_points)
    Z_true = Z_true.reshape(xx.shape)

    # Plot true decision boundaries
    ax.contour(xx, yy, Z_true, levels=[0.5, 1.5], colors=['darkgreen'], linestyles=['-'], label = 'True boundaries')
    

    # Encode the predicted target labels
    le_pred = LabelEncoder()
    df['target_encoded_pred'] = le_pred.fit_transform(df['target_pred'])

    # Fit logistic regression model for predicted targets
    y_pred = df['target_encoded_pred']
    model_pred = LogisticRegression()
    model_pred.fit(X, y_pred)

    # Create a mesh to plot the predicted decision boundaries
    Z_pred = model_pred.predict(grid_points)
    Z_pred = Z_pred.reshape(xx.shape)

    # Plot predicted decision boundaries
    ax.contour(xx, yy, Z_pred, levels=[0.5, 1.5], colors=['purple'], linestyles=['--'], label = 'Predict boundaries')

    # Add custom legend entries for boundaries
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color='darkgreen', linestyle='-', linewidth=2))
    labels.append('True Boundaries')
    handles.append(plt.Line2D([0], [0], color='purple', linestyle='--', linewidth=2))
    labels.append('Predicted Boundaries')
    
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
    
    
    # Show the plot
    plt.show()



def plot_combined_targets_on_image_2(df, img):
    # img_width, img_height = img.size

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img, extent=[df['x'].min(), df['x'].max(), df['y'].min(), df['y'].max()])

    # Plot points based on target values
    colors = {1: 'lightblue', 2: 'plum', 3: 'tomato'}
    
    # Plot each unique coordinate for each target
    for target in colors.keys():
        target_points = df[df['target'] == target]
        ax.scatter(target_points['x'], target_points['y'], 
                   color=colors[target], 
                   label=f'True Target {target}', 
                   s=10,
                   alpha = 0)

    # Set the axis limits to match the coordinate range
    ax.set_xlim([df['x'].min(), df['x'].max()])
    ax.set_ylim([df['y'].min(), df['y'].max()])

    # Encode the true target labels
    le_true = LabelEncoder()
    df['target_encoded_true'] = le_true.fit_transform(df['target'])

    # Fit logistic regression model for true targets
    X = df[['x', 'y']]
    y_true = df['target_encoded_true']
    model_true = LogisticRegression()
    model_true.fit(X, y_true)

    # Create a mesh to plot the true decision boundaries
    x_min, x_max = df['x'].min() - 1, df['x'].max() + 1
    y_min, y_max = df['y'].min() - 1, df['y'].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['x', 'y'])
    Z_true = model_true.predict(grid_points)
    Z_true = Z_true.reshape(xx.shape)

    # Plot true decision boundaries
    ax.contour(xx, yy, Z_true, levels=[0.5, 1.5], colors=['darkgreen'], linestyles=['-'], label = 'True boundaries')
    

        # Plot each unique coordinate for each target
    for target in colors.keys():
        target_points = df[df['target_pred'] == target - 1]
        ax.scatter(target_points['x'], target_points['y'], 
                   color=colors[target], 
                   label=f'Predicted Target {target}', 
                   s=10,
                   alpha = 0.8)
    # Encode the predicted target labels
    

    # Add custom legend entries for boundaries
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color='darkgreen', linestyle='-', linewidth=2))
    labels.append('True Boundaries')

    
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
    
    
    # Show the plot
    plt.show()