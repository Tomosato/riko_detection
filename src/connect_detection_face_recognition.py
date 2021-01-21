import cv2
import numpy as np
from PIL import Image

from face_detection import FaceRecognition
from human_clipping import get_bounding_box


def opencv2pillow(image):
    new_image = image.copy()
    new_image = Image.fromarray(new_image)
    return np.array(new_image)


def connect(image_paths, csv_path, leran_path):
    """
    物体検出の結果を顔認証のエンジンに渡す

    image_paths: 対象の画像のパスのリスト
    csv_path: 訓練データリストの入っているcsvファイルのパス
    learn_path: 顔認証に必要な写真が入っているディレクトリのパス
    """
    target_image_imfos = get_bounding_box(image_paths)

    for image_path, candidate_list in target_image_imfos.items():
        loaded_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
        candidate_images = []
        for image_points in candidate_list:
            # 小数点はだめなので、妥協の産物としてint型に変換しておく
            x = int(image_points[0])
            y = int(image_points[1])
            w = int(image_points[2])
            h = int(image_points[3])

            new_image = opencv2pillow(loaded_image[y:y + h, x:x + w])
            candidate_images.append(new_image)

        # 顔認証
        face_recognition = FaceRecognition()
        learn_df = face_recognition.get_train_file_DataFrame(csv_path, leran_path)

        for number, candidate_image in enumerate(candidate_images):
            compare_result = face_recognition.detection_riko(learn_df, candidate_image)

            if compare_result:
                save_name = "{}_result_{}.jpg".format(image_path, number)
                save_target_image = Image.fromarray(candidate_image)
                save_target_image.save(save_name, quality=95)
                print("{} saved.".format(save_name))
            else:
                print("Image not saved.")
