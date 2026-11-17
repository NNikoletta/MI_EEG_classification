# Classification of Motor Imagery EEG signals using neural networks
 
This repository contains the code and results of the MI EEG classification research project.
 
## Status
 
Completed.
 
## Tech stack
 
Python, TensorFlow/Keras, MNE, NumPy, pandas, scikit-learn, Matplotlib, seaborn
 
## Abstract
 
Systems that are able to create a connection between the brain and an external device are called Brain-Computer Interfaces and they have been an actively researched topic for many years. Such systems enable humans to interact with their environment in a new way, by only using their thoughts. BCIs could be used for aiding individuals with motor impairments, such as locked-in syndrome (LIS) and tetraplegia, among others.
 
The key element of any BCI is translating the thoughts, brain signals that occur due to the electric activity in the brain of an individual into commands that then can be used for controlling various types of devices, such as a robotic arm, or a cursor on a screen. The classification of EEG signals is a part of the translation process.
 
Classification of Motor Imagery (MI) EEG signals- signals that occur when an individual imagines the movement of a certain body part- is a challenging task. Due to the nature of the signals, they are extremely difficult to decode. Brain signals are very complex, and while the temporal resolution of the EEG is quite high, the spatial resolution is poor. One of the methods for deciphering and classifying the EEG data that has gained a lot of popularity in the past years are Neural Networks. They are inspired by the human brain and are able to find patterns and underlying connections in the provided data.
 
The aim of this study was to gain a good understanding of BCIs, EEG signals, and neural networks as a method used for classifying MI EEG signals, as well as to use the acquired knowledge to design and implement networks for four-class MI EEG classification tasks. In this work, a multi-branch attention-based convolutional neural network called MultiConvNet is proposed. The model consists of three threads the network branches into after the first convolutional layer. These are responsible for the extraction of different features as well as for squeezing the channel dimension which reduces the size of the data, thus making the computations ahead faster. A Temporal Attention Module is applied to the data of one of the branches to highlight the information of interest along the time axis. The features extracted by the branches are forwarded to a Feature Attention Module, which enhances the features that are relevant to the classification task. An average pooling, a fully connected layer and a classifier follows the layers described above. The network was evaluated on PhysioNet’s publicly available EEG Motor Movement/Imagery Dataset using five-fold cross-validation, and it achieved the accuracy of 57.89% with the runtime of 22.37 minutes. This network performs as well as other state-of-the-art models but requires a shorter training period. In addition to the MultiConvNet, another network, called MultiConvNetEEG is proposed in this work. It was created by adding an EEGNet block in a form of a branch to the MultiConvNet. Various popular architectures were trained and tested, and their performance was compared to the proposed MultiConvNetEEG, which achieved the accuracy of 59.06% with the runtime of 52.23 minutes. A Wilcoxon signed-rank test was performed to evaluate the differences in performance, based on the results of which, the MultiConvNetEEG performed statistically better than the tested ShallowConvNet, EEGNet, MBEEGCBAM and the DSCNN architectures.
 
## Dataset
 
### PhysioNet MI Dataset
 
In this research, the created models were evaluated on PhysioNet’s publicly available EEG Motor Movement/Imagery Dataset that was recorded by the developers of the BCI2000 instrumentation system.
 
The dataset contains the EEG signals of 109 subjects, the recordings are one, or two minutes long and their number exceeds 1500. The EEG signals were recorded using 64 electrodes and sampled at 160Hz; each subject took part in fourteen measurements. The first two measurements were one-minute-long baseline runs, during one the eyes of the subject were opened, during the other closed. After the baseline runs the subject had to perform four different tasks three times in their initial order, the result was twelve two-minute-long measurements. Overall, each subject took part in fourteen recordings, during which they had a screen before them where a target corresponding to the movement they had to perform or imagine appeared. The four tasks were the following:
 
1. Opening and closing of the left or right fist when the target appears on the left or right side of the screen, respectively. The task must be performed until the target disappears, after the subject can relax.
2. Imagining the opening and closing of the left or right fist when the target appears on the left or right side of the screen, respectively. The task must be performed until the target disappears, after the subject can relax.
3. Opening and closing of both fists or both feet when the target appears on the top or bottom of the screen, respectively. The task must be performed until the target disappears, after the subject can relax.
4. Imagining the opening and closing of both fists or both feet when the target appears on the top or bottom of the screen, respectively. The task must be performed until the target disappears, after the subject can relax.
The data was originally put into one of the following three categories: T0- rest, T1- opening and closing either the left or right fist, T2- opening and closing either both fists or both feet.
 
### Loading and pre-processing
 
In this work only the measurements of the imagined movements were used for classification, these are the Motor Imagery EEG signals, therefore, instead of fourteen recordings, only six were needed, these are the following: 4, 6, 8, 10, 12, 14. Furthermore, the EEG signals of subjects 88, 89, 92, and 100 were excluded.
 
The original three classes (T0- rest, T1- opening and closing either the left or right fist, T2- opening and closing either both fists or both feet) were modified. The data was changed to a four-class classification-problem. First the rest class was deleted, then both remaining classes were divided into two more. The updated dataset, thus, consisted of the following four classes:
 
- 0: opening and closing the left fist
- 1: opening and closing the right fist
- 2: opening and closing both fists
- 3: opening and closing both feet
The dataset was formed by using 2-seconds-long input segments. The epochs were created starting at -0.1 before the event. Before using the data, it was normalized, scaled to a 0 to 1 range.
 
### Training and testing
 
Different functions were implemented for creating the sets of data used for training, validation, and testing. The first created method for building the sets loads the data of a given number of subjects. The entirety of the data is then divided into three datasets using the train_test_split function. 33% of all data is used for testing, 80% of the remaining records are used for training and the rest for validation. This method, however, is not used in the evaluation process, since some of the records of one individual could belong to the train and some to the test set. To provide an accurate evaluation of the models’ performance, the networks have to be tested on a dataset that consists of data and features it has not encountered before.
 
The second method creates datasets by subjects. It uses the data of a given number of subjects to create the train set and the data of another given number of subjects for the validate set. 80% of the train data is actually used for training, and the rest of the data is assigned to the validate set.
 
The third method is similar to the second one, but it prepares the data for cross-validation. In this work, five-fold cross-validation was used for evaluating the performance of the models, a brief description of the technique follows. The whole dataset (records of 105 subjects) is divided into five subgroups, each consisting of data of twenty-one individuals, the training set is created from the data of four folds (80% of data used for training, and 20% for validation), the remaining fold is used for testing. The model is then trained and tested five times, each time using a different subset of the folds. The achieved accuracy and loss are recorded, and the mean is calculated, which is used as one of the characteristics of the model’s performance.
 
Each implemented model was first evaluated on the data of the first six subjects using the second method. The train set consisted of the data of the first four subjects. This step was used for determining how the network performs on limited amount of data. After this, using the second method, the models were evaluated on the entire dataset (data of the first 70 subjects was used for training, the test set consisted of the remaining data). Later a five-fold cross-validation was performed. During the evaluation process categorical cross entropy was the loss function, and Adam optimizer was used. The batch size was set to 16 in all cases and each cross-validation subset was trained for 10 epochs. In case of already existing networks, such as the EEGNet, the first two steps were not essential, however, when implementing a new network, the first two steps were crucial in the process of determining which parameters needed adjusting. The main characteristics that describe the performance of the networks are the mean accuracy, mean loss, and the runtime. The cross-validation was performed on a CPU using the ITK PPKE High Performance Computing (HPC) cluster.
 
## Results
 
The performance of a network can be evaluated based on different statistics. In this work three characteristics are used to examine the performance of the models. These are the accuracy, standard deviation, and runtime. Based on accuracy only, the best performance was achieved by the proposed MultiConvNetEEG model. This model provides better classification accuracy on PhysioNet’s dataset than the ShallowConvNet, EEGNet, Two-branch EEGNet, MBEEGCBAM, and DSCNN. The difference in the accuracy is in favour of the proposed model by 3.37%, 1.97%, 1.45%, 1.19%, 2.12%, respectively. While the accuracy achieved by this model is also higher than the accuracy of the three-branch EEGNet, the difference is only 0.85%, thus it is not significantly better.
 
Based on the standard deviation the three-branch EEGNet performed the best. The standard deviation is an important statistic, since it can tell how well a network performs on different sets of data. Due to the nature of the information used in EEG classification, a network’s generalization ability is a crucial characteristic. The second best in this case is the proposed MultiConvNetEEG. When examining the runtime, the best performance was achieved by the proposed MultiConvNet.
 
A visualization of the three characteristics of each architecture is shown in figure 1.
 
A detailed summary of the results is provided in Table 2.
 
In addition to the characteristics described above, confusion matrices were used to assess the performance of the models. Based on these, the details of the networks’ choices can be viewed.
 
Overall, the best performance was achieved by the two proposed models: the Multi ConvNet and MultiConvNetEEG. The former’s advantage is that it is able to provide an accuracy as good as the EEGNet’s and other state-of-the-art models’ that were tested in this study, while keeping the training time short (one epoch was around 25s-long). The MultiConvNetEEG model is able to provide a better accuracy than the simple MultiConvNet but with the cost of a longer runtime. Furthermore, the difference in the performance of the proposed model and the three-branch EEGNet is not significant, thus the models could be used interchangeably.
 
To further investigate the MultiConvNetEEG networks’ performance Wilcoxon signed-rank test was performed, the calculated p-values are shown in Table 1. The results of two five-fold cross-validations were used to calculate the p-values. In case of the ShallowConvNet, EEGNet, MBEEGCBAM, and DSCNN the value is <0.05, this means that the proposed network performs statistically better.
 
<img width="627" height="37" alt="wilcoxon" src="https://github.com/user-attachments/assets/b2524fbd-2563-47a5-98fe-64269171654a" />

Table 1: Results of the Wilcoxon signed-rank test performed between the proposed MultiConvNetEEG and other networks.
 
<img width="2633" height="1892" alt="resultsoverview" src="https://github.com/user-attachments/assets/16b91d18-d1ab-45d7-a401-0059be447cd4" />

Figure 1: The achieved accuracy and runtime of each model.
 
<img width="637" height="242" alt="resultsbymodel" src="https://github.com/user-attachments/assets/fc441b2e-cb4a-4b65-9336-97968398f5fc" />

Table 2: Summary of the evaluation results by model. The best results are shown in bold.
 
## Thesis
 
The full thesis is available upon request.
 
