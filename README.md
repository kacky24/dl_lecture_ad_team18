# dl lecture advanced team18
顔加工アプリの加工を取り除くアプリケーション

## How to use
### Demo
- sh gan/pretrained_models/get_pretrained_model.shで学習済みのパラメータをダウンロード
- demo.ipynbを実行(sample_images内の画像のノイズを除去、可視化する)

### train
- sh dataset/get_data_shでデータをダウンロード
- python gan/train_pix2pix.py or python gan/train_idcgan.py

## Requirement
- python 3.5.3
- numpy 1.13.3
- matplotlib 2.0.2
- torch 0.2.0
- torchvision 0.1.9
- opencv-python 3.3.1.11
