# Emotion recognition with CNN

# DO NOT USE: Currently I think the code is not even running and I really don't have time to fix it

This repository is the out project about mood recognition using convolutional neural network for the course Seminar Neural Networks at TU Delft.

![Angry Test](https://raw.githubusercontent.com/isseu/emotion-recognition-neural-networks/master/paper/angry.png)

 67% Accuracy

 ![Angry Test](https://raw.githubusercontent.com/isseu/emotion-recognition-neural-networks/master/paper/matrix_final.png)

## Dataset

We use the [FER-2013 Faces Database](http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=main), a set of 28,709 pictures of people displaying 7 emotional expressions (angry, disgusted, fearful, happy, sad, surprised and neutral). **The dataset quality and image diversity is not very good and you will probably get a model with bad accuracy in other applications!**

You have to request for access to the dataset or you can get it on [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). Download `fer2013.tar.gz` and decompress `fer2013.csv` in the `./data` folder.

Install all the dependencies using `virtualenv`.

```bash
virtualenv -p python3 ./
source ./bin/activate
pip install -r requirements.txt
```

The data is in CSV and we need to transform it using the script `csv_to_numpy.py` that generates the image and label data in the `data` folder.

```bash
$ python3 csv_to_numpy.py
```

By default this is using AlexNet architectures, but in the paper we propose different ones.

## Usage

```bash
# To train a model
$ python3 emotion_recognition.py train
# To use it live
$ python3 emotion_recognition.py poc
```

## Paper

[Link](https://github.com/isseu/emotion-recognition-neural-networks/blob/master/paper/Report_NN.pdf)

## Contributor
[Ayesha Batool](https://github.com/ayeshabatool119)