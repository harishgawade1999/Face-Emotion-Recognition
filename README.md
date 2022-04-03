
![Live facial](https://user-images.githubusercontent.com/88233990/161426973-78f31969-4547-47f3-b353-768d2c6e339a.jpg)

# Real-Time-Face-Emotion-Recognition

Our Indian Education system are changing rapidly from last some years and specially digital learning platforms or E-learning platforms.
physical classroom a lecturing teacher can see the faces and assess the emotion of the class and tune their lecture accordingly. But when we are talking about digital classrooms,it becomes impossible.So to solve this problem in this project,we have tried to build a real time Face Emotion Recognition system to monitor student's mood during class activity.

# Table-of-contents

- Real-Time-Face-Emotion-Recognition
- Table of contents
- Dependencies
- Installation
- Usage
- Model Creation
- Deployment
- Demo-Preview
- Conclusion
- License

# Dependencies

- TensorFlow
- OpenCV
- Streamlit
- Streamlit webRTC
- Heroku

# Installation

To use this project, first clone the repo on your device using the command below:

```git init```

```git clone ```

```cd Real-Time-Face-Emotion-Recognition```

```Install dependencies and run app.py```

```pip3 install -r requirements.txt```

```streamlit run app.py```

# Usage

This webapp can help the teachers to analyze the students emotions and adjust their teaching strategies accordingly to improve the efficiency of online teaching.

# Model Creation

## Deepface

Deepface is a lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for python. DeepFace is a deep learning facial recognition system created by a research group at Facebook.The program employs a nine-layer neural network with over 120 million connection weights and was trained on four million images uploaded by Facebook users.

![deep](https://user-images.githubusercontent.com/88233990/161428282-97624e4b-ad7f-4329-a308-9c04ccdd358d.jpg)

### Model Evaluation

![Deepface](https://user-images.githubusercontent.com/88233990/161428481-5a28e0e6-1e09-4c50-a964-e16ab243666c.png)

## Simple CNN 

After using deepface, we tried to build the simple CNN model. The model type that we used is Sequential. Sequential is the easiest way to build a model in Keras. It allows you to build a model layer by layer. In this model, we created a three layered CNN Model, two CNN layers and one fully connected layer.

### Model Evaluation

![simple_cnn_accuracy](https://user-images.githubusercontent.com/88233990/161428663-a4f4e584-7bb7-4b89-b58d-46611b7395bb.png)


![simple_cnn_loss](https://user-images.githubusercontent.com/88233990/161428676-845ec2d0-162f-4ff7-b07a-7885711f707f.png)

# Deployment

We have created a front end using streamlit for web app.Used streamlit-webrtc which helped to deal with real time video streams.Image captured from the webcam is sent to the Video transform function to detect emotions.We deployed the model on the streamlit sharing cloud platform.

Streamlit Link - https://share.streamlit.io/harishgawade1999/face-emotion-recognition/main/app.py

# Demo-Preview

https://user-images.githubusercontent.com/88233990/161428825-568ae5ed-f4c2-41c0-9fc0-a4677c0aa4dc.mp4

# Conclusion

Using a deep learning model based on the architecture of CNN, we constructed a framework to analyze student’s emotions. Simple CNN model gave 70.24% train accuracy and 61.06% validation accuracy but was not able to predict emotions precisely when tested in real time. 
CNN using deep layers has 73.27% train accuracy and 58.55% validation accuracy. This model was accurate in predicting tests in real time. There were less disgust images in the dataset that's why the model was confused in detecting disgust. This model can help the teachers to adjust their teaching strategies accordingly to improve the efficiency of online teaching.

# Licence

https://opensource.org/licenses/GPL-3.0





