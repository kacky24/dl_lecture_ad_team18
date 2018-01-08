# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from PIL import Image

def stamp_on_face():
    
    files = os.listdir('../dataset/original')
    
    # 上書きする画像の読み込み
    ol_imgae_path = "../dataset/stamp/stamp_01.png"    
    ol_image = cv2.imread(ol_imgae_path,cv2.IMREAD_UNCHANGED)   # アルファチャンネル(透過)も読みこむようにIMREAD_INCHANGEDを指定
    
    for file in files:

        # 認識対象ファイルの読み込み
        image_path = "../dataset/original/" + file
        image = cv2.imread(image_path)

        # グレースケールに変換
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 顔認識用特徴量のファイル指定
        cascade_path = "./haarcascade_frontalface_alt.xml"
        # カスケード分類器の特徴量を取得する
        cascade = cv2.CascadeClassifier(cascade_path)

        # 顔認識の実行
        facerecog = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

        if len(facerecog) > 0:
            # 認識した顔全てに画像を上書きする
            for rect in facerecog:

                # 認識範囲にあわせて画像をリサイズ
                resized_ol_image = resize_image(ol_image, rect[2], rect[3])
                
                # 画像の上書き作成
                image = overlayOnPart(image, resized_ol_image, rect[0], rect[1])

            
            #　画像の明るさを変える
            # ガンマ定数の定義
            gamma = 1.2
            # ルックアップテーブルの生成
            look_up_table = np.ones((256, 1), dtype = 'uint8' ) * 0
            for i in range(256):
                look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
            # ガンマ変換後の出力
            image = cv2.LUT(image, look_up_table)

        # 認識結果の出力
        cv2.imwrite("../dataset/snow/" + file , image)

def overlayOnPart(src_image, overlay_image, posX, posY):

    # オーバレイ画像のサイズを取得
    ol_height, ol_width = overlay_image.shape[:2]

    # OpenCVの画像データをPILに変換
    #　BGRAからRGBAへ変換
    src_image_RGBA = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    overlay_image_RGBA = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2RGBA)

    #　PILに変換
    src_image_PIL=Image.fromarray(src_image_RGBA)
    overlay_image_PIL=Image.fromarray(overlay_image_RGBA)

    # 合成のため、RGBAモードに変更
    src_image_PIL = src_image_PIL.convert('RGBA')
    overlay_image_PIL = overlay_image_PIL.convert('RGBA')

    # 同じ大きさの透過キャンパスを用意
    tmp = Image.new('RGBA', src_image_PIL.size, (255, 255,255, 0))
    # 用意したキャンパスに上書き
    tmp.paste(overlay_image_PIL, (posX, posY), overlay_image_PIL)
    # オリジナルとキャンパスを合成して保存
    result = Image.alpha_composite(src_image_PIL, tmp)
    result_np=np.asarray(result)
    print(result_np.shape)
    print(result_np)
    return  cv2.cvtColor(result_np[:,:,0:3], cv2.COLOR_RGB2BGR)
       
def resize_image(_image, _height, _width):

    # 元々のサイズを取得
    org_height, org_width = _image.shape[:2]

    # 大きい方のサイズに合わせて縮小
    if float(_height)/org_height > float(_width)/org_width:
        ratio = float(_height)/org_height
    else:
        ratio = float(_width)/org_width

    # リサイズ
    resized = cv2.resize(_image,(int(org_height*ratio),int(org_width*ratio)))

    return resized   
        
if __name__ == '__main__':
    stamp_on_face()