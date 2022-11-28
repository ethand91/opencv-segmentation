import argparse
import cv2
import numpy as np
import mediapipe as mp

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--person", required = True, help = "path to image file of person")
    ap.add_argument("-b", "--background", required = True, help = "path to background image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["person"])
    background_img = cv2.imread(args["background"])

    resized_bg_img = cv2.resize(background_img, (image.shape[1], image.shape[0]))

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segment = mp_selfie_segmentation.SelfieSegmentation(model_selection = 0)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = segment.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_seg_mask = results.segmentation_mask

    threshold = 0.5
    binary_mask = image_seg_mask > threshold

    mask3d = np.dstack((binary_mask, binary_mask, binary_mask))

    replaced_img = np.where(mask3d, image, resized_bg_img)

    cv2.imwrite("result.jpg", replaced_img)
