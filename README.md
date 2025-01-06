# Project_in_Data_Science
## From Pixels to material properties: machine learning in solar cell research
### Introduction:
The rapid development of solar cell technologies necessitates the exploration of new materials, a process that involves a multitude of variables and extensive experimental efforts. This project aims to streamline this process by contributing to “self-driving lab” (SDL) designed to study solar cell materials. The SDL integrates automation with advanced AI techniques to fabricate and analyze new samples efficiently. Central to this SDL is the application of machine learning (ML) methods to identify and characterize phase boundaries within images of material samples. These samples, simulated for the purposes of this project, consist of circular substrates where two chemical elements, such as tin (Sn) and barium (Ba), are deposited in smooth gradients from opposite sides. Additionally, a third element, sulfur (S), is uniformly incorporated, leading to the formation of various phases (chemical compounds) depending on experimental conditions. The main challenge lies in accurately identifying the distribution of these phases by analyzing images of the samples.

Phases with distinct optical properties are detectable through reflected and transmitted light images, and their boundaries are defined by specific compositions. This project leverages ML to detect these phase boundaries swiftly and accurately, facilitating a comprehensive understanding of the material’s phase diagram. By utilizing simulated images with known phase boundary positions, the project aims to train ML models capable of extracting relevant information and efficiently constructing phase diagrams. These advancements will significantly enhance the ability to explore and optimize new materials for solar cells, reducing the need for labor-intensive experimental procedures and accelerating the development of more efficient solar technologies.
### Results:
1. The test accuracy of the Multinomial Logistic Regression model ranges from 80% to 81%.
2. The results using CNNs were significantly better compared to the simple model. Specifically, the test accuracy improved to 91.97%.
3. The U-Net modeling results achieved a high test accuracy of 97.13%.
   
### Contributed by Aiman
### Contributed by Ruizhen Chen
#### Data Preprocessing Steps
1. In the data preprocessing phase, I established a data pipeline to clean the data and split the dataset. Finally, I experimented with three models as baseline models: Logistic Regression, Decision Tree, and K-Nearest Neighbors. Among them, the accuracy of Logistic Regression was 80.7%, which is lower than the results achieved by Yagna Karthik and Khang.
2. The code can be found at: [GitHub Repository](https://github.com/Karthik1000/Project_in_Data_Science/blob/Final_code_submission/data_preprocessing.ipynb) with my comments and explanations.
#### U-Net  model
1. Yagna Karthik and I collaborated on building the U-Net model, experimenting with various epochs and accuracies, and testing it under different conditions, including varied train-validation-test data splits and batch sizes.
2. Developed a data processing pipeline that includes image cropping, enhancement, and coordinate normalization, aligning the CSV data with the image data to ensure consistency in the input data.
3. Implemented a generalized ‘DataGenerator’ class that supports batch data loading, data augmentation, and dynamic label generation, enhancing the efficiency of data processing and the training process.
4. Designed a U-Net-based architecture adapted for multi-channel input and incorporated a multi-class Softmax activation function. Additionally, techniques such as dynamic learning rate adjustment were employed to prevent model overfitting. The initial model achieved an accuracy of 94.4% on the test set.
5. The code can be found at: [GitHub Repository](https://github.com/Karthik1000/Project_in_Data_Science/blob/Final_code_submission/UNet_without_contour.ipynb) with my comments and explanations.
#### Final report
I contributed to the final report by writing the sections in Parts IV and V about the U-Net model, providing a detailed explanation of the U-Net architecture and the structure of its inputs and outputs. Additionally, I presented a comparison of true boundaries versus predicted boundaries (without contours) in the report. Yagna Karthik further enhanced this by adding advanced visualizations with contours.
### Contributed by Khang
#### Data Preprocessing Steps
1. I do not have much experience with data pipelines, so I did not implement the code for this part. However, I participated in group meetings and contributed ideas to this section.
2. Yagna Karthik initially worked on the Logistic Regression model (a simple model). I later joined him, and together we achieved a test accuracy of 82%.
#### Image Preprocessing Steps
I proposed the idea for a function to plot true and predicted boundaries and implemented it.
#### CNNs model
I worked on this model independently and compared my results with the U-Net model implemented by Ruizhen and Yagna Karthik. In summary:
1. I transformed the tabular data, using the three columns Sn, Ba, and F, into image data.
2. I used CNNs with 9 channels: 3 channels from the transformed tabular data, 3 channels from transmitted image data, and 3 channels from reflected image data. In this step, I used the enhanced image function implemented by Yagna Karthik. Aside from the enhanced image function, I implemented all other parts myself.
3. I calculated the test accuracy for the CNNs model and plotted the predicted phase boundaries.
4. The code can be found at: [GitHub Repository](https://github.com/Karthik1000/Project_in_Data_Science/blob/Final_code_submission/ProjectDS_Khang_CNNs.ipynb) with my comments and explanations.
#### Final report
I contributed to writing the final report, including the initial drafts of Parts 1, 2, 3, and 4 (excluding the U-Net model), and Part 5 (excluding the U-Net model results). After reviews by Yagna Karthik, Ruizhen, and two advisors, I continued refining and modifying the final report.
### Contributed by Yagna Karthik Vaka
#### Data Preprocessing Steps
1. I have contributed to these data-preprocessing steps and started initially. Khang has made a few modifications to these by changing the target columns and reduced to 1 instead of 3 columns.
2. We both have worked on these steps and achieved an accuracy of 82% on test accuracy.
3. I have added all comments and explanations of the [code]( https://github.com/Karthik1000/Project_in_Data_Science/blob/Final_code_submission/Project_DS_Data_UNet_model_final.ipynb) 
#### Unet model
1. In this model, Ruizchen and I have worked together to build the model and tested with different epochs and accuracies and also tested with different combinations of train-validation-test data split and with different batches.
2. Refined contour visualizations to address alignment issues on the right side of the plot, improving clarity and precision.
3. Highlighted areas of strong separation between true and predicted labels using black markers, aiding in the identification of model performance strengths and weaknesses.
4. Finally, I have used a data split of 100-50-50 for train-val-test splits and achieved a test accuracy of 97.13%.
5. I have added all comments and explanations of the [code]( https://github.com/Karthik1000/Project_in_Data_Science/blob/Final_code_submission/Project_DS_Data_UNet_model_final.ipynb)
#### Final report
I contributed to the final report by writing sections in Parts II, IV, and V, and adding images related to the U-Net model in the results and discussion section. Additionally, I proofread the entire document, ensuring consistency and alignment with the project's objectives. This included updating details about the number of data points and refining the train-validation-test split ratios for accuracy and clarity. Other team members like Khang and Ruizchen have contributed to most of the above Parts in more detailed.
