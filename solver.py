from keras.models import load_model
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
from PIL import Image, ImageOps

def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    (h, w) = image.shape[:2]

    if w > h:
        image = imutils.resize(image, width=width)

    else:
        image = imutils.resize(image, height=height)

    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image


class solver:
    MODEL_FILENAME = "captcha_solver\captcha_model.hdf5"
    MODEL_LABELS_FILENAME = "captcha_solver\model_labels.dat"

    def __init__(self):
        with open(self.MODEL_LABELS_FILENAME, "rb") as f:
            self.lb = pickle.load(f)

        self.model = load_model(self.MODEL_FILENAME)

    def solve(self, image_file):
        name = open(image_file).name
        ImageOps.expand(Image.open(image_file), border=5, fill='white').save(name)
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = contours[1] if imutils.is_cv3() else contours[0]

        letter_image_regions = []

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if w / h > 1.25:
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                letter_image_regions.append((x, y, w, h))

        if len(letter_image_regions) != 6:
            raise Exception("can't recognize text")

        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        output = cv2.merge([image] * 3)
        predictions = []

        for letter_bounding_box in letter_image_regions:
            x, y, w, h = letter_bounding_box

            letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
            try:
                letter_image = resize_to_fit(letter_image, 20, 20)
            except Exception:
                continue

            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            prediction = self.model.predict(letter_image)

            letter = self.lb.inverse_transform(prediction)[0]
            predictions.append(letter)

            cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
            cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        captcha_text = "".join(predictions)

        return captcha_text

        # print(f"Текст: {captcha_text}")

        # cv2.imshow("Output", output)
        # cv2.waitKey()

