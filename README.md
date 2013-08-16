Neu_Script
==========

The ready-to-use implementation of multilayered neural network. Some modules used the skeleton developed by UFLDL.

It has sofmax and fine-tune developped for the sake of classification. It has already achieved 100% accuracy on MNIST dataset. I am currently using this tool to run expirements on biomedical brain images, in terms of early diagnosis(preprocessed features).

To start to train the network and test it. pls start with ./deepTrain/godeep.m as an example.

If you are interested in further development to gain some enhances. Please contact me (lsqshr@gmail.com).
What can be improved?
1. coding styles.
2. matlab gui app
3. add more gpu use for matrix multiplication. I have not add any gpu utilisation, because for my biomedical dataset, I am worrying that the transferring overhead will make the enhancement not very obvious. However, if you are looking to apply deep learing on raw image processing, you are welcome to enhance the gpu utilization. 

I am happy to share the code with any one. You can even also take my code as a reference, because not all the descriptions in UFLDL tutorials are clear to follow, such as the fine-tune section.
