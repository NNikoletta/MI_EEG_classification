# Classification of Motor Imagery EEG signals using neural networks
This repository contains the code and results of the MI EEG classification research project.

# Status
Completed.

# Abstract
Systems that are able to create a connection between the brain and an external device
are called Brain-Computer Interfaces and they have been an actively researched topic for
many years. Such systems enable humans to interact with their environment in a new
way, by only using their thoughts. BCIs could be used for aiding individuals with motor
impairments, such as locked-in syndrome (LIS) and tetraplegia, among others.
The key element of any BCI is translating the thoughts, brain signals that occur due
to the electric activity in the brain of an individual into commands that then can be used
for controlling various types of devices, such as a robotic arm, or a cursor on a screen.
The classification of EEG signals is a part of the translation process.
Classification of Motor Imagery (MI) EEG signals- signals that occur when an indi
vidual imagines the movement of a certain body part- is a challenging task. Due to the
nature of the signals, they are extremely difficult to decode. Brain signals are very com
plex, and while the temporal resolution of the EEG is quite high, the spatial resolution is
poor. One of the methods for deciphering and classifying the EEG data that has gained a
lot of popularity in the past years are Neural Networks. They are inspired by the human
brain and are able to find patterns and underlying connections in the provided data.
The aim of this study was to gain a good understanding of BCIs, EEG signals, and
neural networks as a method used for classifying MI EEG signals, as well as to use the ac
quired knowledge to design and implement networks for four-class MI EEG classification
tasks. In this work, a multi-branch attention-based convolutional neural network called
MultiConvNet is proposed. The model consists of three threads the network branches
into after the first convolutional layer. These are responsible for the extraction of different
features as well as for squeezing the channel dimension which reduces the size of the data,
thus making the computations ahead faster. A Temporal Attention Module is applied to
the data of one of the branches to highlight the information of interest along the time
axis. The features extracted by the branches are forwarded to a Feature Attention Mod
ule, which enhances the features that are relevant to the classification task. An average
pooling, a fully connected layer and a classifier follows the layers described above. The
network was evaluated on PhysioNet’s publicly available EEG Motor Movement/Imagery
Dataset using five-fold cross-validation, and it achieved the accuracy of 57,89% with the
runtime of 22,37 minutes. This network performs as well as other state-of-the-art models
but requires a shorter training period. In addition to the MultiConvNet, another net
work, called MultiConvNetEEG is proposed in this work. It was created by adding an
EEGNet block in a form of a branch to the MultiConvNet. Various popular architectures
were trained and tested, and their performance was compared to the proposed Multi
ConvNetEEG, which achieved the accuracy of 59,06% with the runtime of 52,23 minutes.
A Wilcoxon signed-rank test was performed to evaluate the differences in performance,
based on the results of which, the MultiConvNetEEG performed statistically better than
the tested ShallowConvNet, EEGNet, MBEEGCBAM and the DSCNN architectures.

# Results
Results
The performance of a network can be evaluated based on different statistics. In this
work three characteristics are used to examine the performance of the models. These are
the accuracy, standard deviation, and runtime. Based on accuracy only, the best per
formance was achieved by the proposed MultiConvNetEEG model. This model provides
better classification accuracy on PhysioNet’s dataset than the ShallowConvNet, EEGNet,
Two-branch EEGNet, MBEEGCBAM, and DSCNN. The difference in the accuracy is in
favour of the proposed model by 3,37%, 1,97%, 1,45%, 1,19%, 2,12%, respectively. While
the accuracy achieved by this model is also higher than the accuracy of the three-branch
EEGNet, the difference is only 0,85%, thus it is not significantly better.
Based on the standard deviation the three-branch EEGNet performed the best. The
standard deviation is an important statistic, since it can tell how well a network performs
on different sets of data. Due to the nature of the information used in EEG classification,
a network’s generalization ability is a crucial characteristic. The second best in this case
is the proposed MultiConvNetEEG. When examining the runtime, the best performance
was achieved by the proposed MultiConvNet.
A visualization of the three characteristics of each architecture is shown in figure 5.1.
A detailed summary of the results is provided in Table 5.2.
In addition to the characteristics described above, confusion matrices were used to
asses the performance of the models (figures 5.2 and 5.3). Based on these, the details of
the networks’ choices can be viewed.
Overall, the best performance was achieved by the two proposed models: the Multi
ConvNet and MultiConvNetEEG. The former’s advantage is that it is able to provide an
accuracy as good as the EEGNet’s and other state-of-the-art models’ that were tested
in this study, while keeping the training time short (one epoch was around 25s-long).
The MultiConvNetEEG model is able to provide a better accuracy than the simple Mul
tiConvNet but with the cost of a longer runtime. Furthermore, the difference in the
performance of the proposed model and the three-branch EEGNet is not significant, thus
the models could be used interchangeably.
To further investigate the MultiConvNetEEG networks’ performance Wilcoxon signed
rank test was performed, the calculated p-values are shown in Table 5.1. The results of
two five-fold cross-validations were used to calculate the p-values. In case of the Shallow
COnvNet, EEGNet, MBEEGCBAM, and DSCNN the value is <0,05, this means that
the proposed network performs statistically better.

<img width="627" height="37" alt="wlicoxon" src="https://github.com/user-attachments/assets/b2524fbd-2563-47a5-98fe-64269171654a" />

Table: Results of the Wilcoxon signed-rank test performed between the proposed
MultiConvNetEEG and other networks.

<img width="2633" height="1892" alt="resultsoverview" src="https://github.com/user-attachments/assets/16b91d18-d1ab-45d7-a401-0059be447cd4" />

Figure: The achieved accuracy and runtime of each model.

<img width="637" height="242" alt="resultsbymodel" src="https://github.com/user-attachments/assets/fc441b2e-cb4a-4b65-9336-97968398f5fc" />

Table: Summary of the evaluation results by model. The best results are shown in
bold.
