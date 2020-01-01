from imageai.Prediction import ImagePrediction
import os
import sys


work_path = os.getcwd()
detector = ImagePrediction()
detector.setModelTypeAsDenseNet()
detector.setModelPath(os.path.join(
    work_path, "DenseNet-BC-121-32.h5"))
detector.loadModel()


def detectMultiPictures(pictures):

    result_array = detector.predictMultipleImages(
        pictures, result_count_per_image=5)

    for result in result_array:
        detection, probabilities = result["predictions"], result["percentage_probabilities"]
        for eachDetection, eachProbabilities in zip(detection, probabilities):
            print(eachDetection, eachProbabilities)
        print('------------------')


def detectSinglePicture(picture):

    detection, probabilities = detector.predictImage(
        picture, result_count=5)
    for eachDetection, eachProbabilities in zip(detection, probabilities):
        print(eachDetection, eachProbabilities)
    print('------------------')


if __name__ == '__main__':
    args = sys.argv
    if len(args) > 2:
        image_list = args[1:]
        detectMultiPictures(image_list)
    else:
        detectSinglePicture(args[1])
