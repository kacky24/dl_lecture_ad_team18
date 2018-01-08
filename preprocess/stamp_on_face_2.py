# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from PIL import Image
import random
# version 2a 目の拡大、検出時のパラメータ調整、スタンプのランダム化　タレント年鑑以外用

global random_case

'''
パスについては適宜変更してください。
haarcascade_frontalface_alt.xml
haarcascade_eye.xml
はOpenCVの中にあるものを使用しています。
'''

# ランダムにスタンプ（のパス）を選ぶ関数
def random_stamp():
    __random_number=random.randint(1,10)
    if (__random_number<10):
        __path="C:/DL2/data/stamp2/stamp_0"+str(__random_number)+".png" 
    else:
        __path="C:/DL2/data/stamp2/stamp_"+str(__random_number)+".png"
    return __path

# 画像の一部にガウシアンフィルタ
def part_of_blur(_img,_roi_blur):
    blur = cv2.GaussianBlur(_img[_roi_blur[1]:_roi_blur[1]+_roi_blur[3],_roi_blur[0]:_roi_blur[0]+_roi_blur[2],:],(5,5),0) 
    _img[_roi_blur[1]:_roi_blur[1]+_roi_blur[3],_roi_blur[0]:_roi_blur[0]+_roi_blur[2],:]=blur

def stamp_on_face():
    def rectangle_func(__image,__roi,__color): # 色を指定して矩形描画
        cv2.rectangle(__image,(__roi[0],__roi[1]),(__roi[0]+__roi[2],__roi[1]+__roi[3]),__color,3)
    
    files = os.listdir('C:/DL2/data/original')
    
    for file in files:
        # 認識対象ファイルの読み込み
        image_path = "C:/DL2/data/original/" + file
        image = cv2.imread(image_path)
        
        if isinstance(image,type(None)):
            continue
        
        # 画像のサイズ、チャンネル数を取得
        image_height,image_width, image_channel = image.shape[:3]
        
        # モノクロ画像は加工しない
        if (image_channel!=3):
            continue
        
        # 検知器の動作確認用
        display = cv2.imread(image_path)

        # グレースケールに変換
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 顔認識用特徴量のファイル指定
        cascade_path = "C:/DL2/haarcascade_frontalface_alt.xml"
        
        # カスケード分類器の用意
        #　顔
        cascade = cv2.CascadeClassifier(cascade_path)
        #　目
        eye_cascade_path="C:/DL2/haarcascade_eye.xml"
        eye_cascade=cv2.CascadeClassifier(eye_cascade_path)

        # 顔認識の実行
        minFaceSize= (int(image_height/8),int(image_width/5)) # 画面に対して顔はそれなりに大きい データセット依存
        facerecog = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=minFaceSize)
        
        if len(facerecog) == 1: # 顔が1個確認されたときのみ処理
            # 上書きする画像の読み込み
            ol_imgae_path = random_stamp()    
            ol_image = cv2.imread(ol_imgae_path,cv2.IMREAD_UNCHANGED)   # アルファチャンネル(透過)も読みこむようにIMREAD_INCHANGEDを指定
 
            # 認識した顔に対して加工をする
            for rect in facerecog:
                # 認識結果の描画
                rectangle_func(display,rect,(0,0,255))
   
                # 認識範囲にあわせて画像をリサイズ
                resized_ol_image = resize_image(ol_image, rect[2], rect[3])                
                
                # 認識した顔の範囲内で目を認識する
                minEyeSize =(int(rect[2]/20),int(rect[3]/50)) #顔に比べてこんなに目は小さくない
                maxEyeSize =(int(rect[2]/2),int(rect[3]/2)) #顔に比べてこんなに目は大きくない
                roi_gray = image_gray[rect[1]:int(rect[1]+0.7*rect[3]), rect[0]:rect[0]+rect[2]] # 目は顔の上の方に存在する
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=1, minSize=minEyeSize, maxSize=maxEyeSize)
                if (len(eyes) == 2 ): # 目が2個確認されたときのみ目拡大
                    print(eyes.shape)
                    for rect_eye in eyes:
                        # 目の元画像内での位置
                        rect_eye_sum = [0,0,0,0]
                        rect_eye_sum[0]=rect_eye[0]+rect[0]
                        rect_eye_sum[1]=rect_eye[1]+rect[1]
                        rect_eye_sum[2]=rect_eye[2]
                        rect_eye_sum[3]=rect_eye[3]
                        # 目の検知された場所を描画
                        rectangle_func(display,rect_eye_sum,(255,0,0))
                        #　目を拡大する範囲を決定する
                        ## OpenCVの目検出器では瞳周辺の眉毛あたりまでを目の矩形として出力するため、一部のみ拡大する
                        rect_eye_expand = [0,0,0,0]
                        rect_eye_expand[0]=rect_eye[0]+rect[0]
                        rect_eye_expand[1]=rect_eye[1]+int(0.3*rect_eye[3])+rect[1]
                        rect_eye_expand[2]=rect_eye[2]
                        rect_eye_expand[3]=int(0.6*rect_eye[3])
                        #　目を拡大する
                        deform_part(image,rect_eye_expand,1.2)
                        '''もし目の拡大量もランダムにしたければどうぞ（スタンプによって拡大量は決まっているので
                        この時点でランダムにするよりかはスタンプを決めた時に拡大量を決めたほうが良い）
                        if flag_eye == 1:
                            deform_part(image,rect_eye_expand,1.2)
                        elif flag_eye == 2:
                            deform_part(image,rect_eye_expand,1.25)
                        else:
                            deform_part(image,rect_eye_expand,1.3)
                        # 目の下をごまかす
                        rect_eye_down = [0,0,0,0]
                        rect_eye_down[0] = rect_eye[0]+rect[0]-5
                        rect_eye_down[1] = rect_eye[1]+rect_eye[3]+rect[1]-5
                        rect_eye_down[2] = rect_eye[2]+10
                        rect_eye_down[3] = 10
                        part_of_blur(image,rect_eye_down)
                        '''
                """        
                # 顎を縮小する -> 以下の実装は無理そうだ。顔の検出部分がファジーなので
                rect_reduct = [0,0,0,0]
                rect_reduct[0]=rect[0]
                rect_reduct[1]=rect[1]+int(0.6*rect[3])
                rect_reduct[2]=rect[2]
                rect_reduct[3]=int(0.4*rect[3])
                deform_part(image,rect_reduct,0.8)
                rect_expand = [0,0,0,0]
                rect_expand[0]=0
                rect_expand[1]=rect[1]+rect[3]
                rect_expand[2]=image_width-1
                rect_expand[3]=image_height-1-rect[1]-rect[3]
                rect_expand_ratio=(rect_expand[3]+int(0.4*rect[3])*0.2)/rect_expand[3]
                deform_part(image,rect_reduct,rect_expand_ratio,_top=0)
                """
                # スタンプを適用する
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
            cv2.imwrite("C:/DL2/data/snow/" + file , image)
            
            # 認識結果の動作確認出力
            cv2.imwrite("C:/DL2/data/display/"+file,display)

# 画像の一部を指定された拡大率でデフォルメする
def deform_part(_image_color,_roi_deform, _magY,_top = 1): # 対象のカラー画像、デフォルメする範囲、縦の拡大率、上揃えか下揃えか    
    # 画像の一部を拡大した画像をつくる
    deform_color = np.copy(_image_color[_roi_deform[1]:_roi_deform[1]+_roi_deform[3], _roi_deform[0]:_roi_deform[0]+_roi_deform[2], :])
    deform_color = cv2.resize(deform_color,(_roi_deform[2],int(_magY*_roi_deform[3])))
    if _top==1:
        _image_color[_roi_deform[1]:_roi_deform[1]+int(_magY*_roi_deform[3]), _roi_deform[0]:_roi_deform[0]+_roi_deform[2], :]=deform_color
    else:
        _image_color[_roi_deform[1]+_roi_deform[3]-int(_magY*_roi_deform[3]):_roi_deform[1]+_roi_deform[3], _roi_deform[0]:_roi_deform[0]+_roi_deform[2], :]=deform_color
        
# 画像の一部を平滑化する
        
# pngの画像を貼りつける
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
    return  cv2.cvtColor(result_np[:,:,0:3], cv2.COLOR_RGB2BGR)
       
def resize_image(_image, _width,_height): # 旧版にはバグあり。

    # 元々のサイズを取得
    org_height, org_width = _image.shape[:2]

    # 大きい方のサイズに合わせて縮小
    if float(_height)/org_height > float(_width)/org_width:
        ratio = float(_height)/org_height
    else:
        ratio = float(_width)/org_width

    # リサイズ
    resized = cv2.resize(_image,(int(org_width*ratio), int(org_height*ratio)))

    return resized   
        
if __name__ == '__main__':
    stamp_on_face()