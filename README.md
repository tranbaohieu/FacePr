## Face recognition with pytorch and dlib
In this example, pytorch is used to implement CNN model inspired by OpenFace project. The model is a variant of the NN4 architecture and identified as nn4.small2 model in the OpenFace project. The model training aims to learn an embedding of an image that the L2 distance between all faces of the same identity is small and the distance between a pair of faces from different identities is large. By selecting suitable threshold, the model can recognize faces in own dataset. Note that this model can run on CPU.
## Pre-requisite
# Install needed packages
```pip install -r requirements.txt```
## Implement
# Prepare images for training
In the folder image there are folders containing images of people that we want to recognize. Each folder has 5 images of a person. If you want to have more people, just create folders and put images inside. It is recommended to have at least 5 images per person and the number of images of each person should be equal.

The images used for training should have only ONE face of the person.

Image file must be in .jpg format.

Run face_detect_and_save.py. It will go through images in folders in the image and detect the face and save it (replace the full image).
# Training and testing
```python main.py```
# Demo
```python demo.py```