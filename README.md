# riko_detection
中山莉子さんを自動検出して切り出す

https://twitter.com/matudaieeeera/status/1342112905618018305

を見て、やってみようとおもった。

# セットアップ方法

pipenvコマンドからのコマンドが依存関係の解消でエラーがはっせいするので

```
pipenv shell
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install cmake face_recognition numpy opencv-python
```

で無理矢理通した