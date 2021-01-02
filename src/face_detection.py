import face_recognition
import cv2
import numpy as np
import glob
import pandas as pd


class FaceRecognition():
    """
    顔の識別を実施するクラス
    """

    def get_train_file_DataFrame(self, csv_path, target_path="../data/face/"):
        """
        学習用の画像のパスを取得して、対応するファイルと人物のデータフレームを作成する

        csvのファイル形式(例)

        ```
        file_name,name
        1.jpg,riko
        2.jpg,mayama
        3.jpg,ayaka
        4.jpg,mirei
        5.jpg,hinata
        6.jpg,kaho
        ```
        """
        train_list = pd.read_csv(csv_path)
        train_list["file_name"] = train_list["file_name"].apply(lambda x: "{}{}".format(target_path, x))

        return train_list

    def detection_riko(self, df, target_image_path):
        """
        画像の情報が入っているデータフレームをもらってきて、その情報
        をもとに学習する
        """
        encoded_face_list = []
        for file_path in df["file_name"]:
            loaded_face_image = face_recognition.load_image_file(file_path)
            encoed_image = face_recognition.face_encodings(loaded_face_image)[0]

            encoded_face_list.append(encoed_image)

        target_image = face_recognition.load_image_file(target_image_path)
        face_locations = face_recognition.face_locations(target_image)
        face_encodings = face_recognition.face_encodings(target_image, face_locations)

        for face_encode in face_encodings:
            matches = face_recognition.compare_faces(encoded_face_list, face_encode)
            face_distances = face_recognition.face_distance(encoded_face_list, face_encode)

            best_match_index = np.argmin(face_distances)

            name = "unkonwn"

            if matches[best_match_index]:
                name = df["name"][best_match_index]

            print(name)