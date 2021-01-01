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

        csvのファイル形式

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
