# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from PIL import Image
import random
# version 2a �ڂ̊g��A���o���̃p�����[�^�����A�X�^���v�̃����_�����@�^�����g�N�ӈȊO�p

global random_case

'''
�p�X�ɂ��Ă͓K�X�ύX���Ă��������B
haarcascade_frontalface_alt.xml
haarcascade_eye.xml
��OpenCV�̒��ɂ�����̂��g�p���Ă��܂��B
'''

# �����_���ɃX�^���v�i�̃p�X�j��I�Ԋ֐�
def random_stamp():
    __random_number=random.randint(1,10)
    if (__random_number<10):
        __path="C:/DL2/data/stamp2/stamp_0"+str(__random_number)+".png" 
    else:
        __path="C:/DL2/data/stamp2/stamp_"+str(__random_number)+".png"
    return __path

# �摜�̈ꕔ�ɃK�E�V�A���t�B���^
def part_of_blur(_img,_roi_blur):
    blur = cv2.GaussianBlur(_img[_roi_blur[1]:_roi_blur[1]+_roi_blur[3],_roi_blur[0]:_roi_blur[0]+_roi_blur[2],:],(5,5),0) 
    _img[_roi_blur[1]:_roi_blur[1]+_roi_blur[3],_roi_blur[0]:_roi_blur[0]+_roi_blur[2],:]=blur

def stamp_on_face():
    def rectangle_func(__image,__roi,__color): # �F���w�肵�ċ�`�`��
        cv2.rectangle(__image,(__roi[0],__roi[1]),(__roi[0]+__roi[2],__roi[1]+__roi[3]),__color,3)
    
    files = os.listdir('C:/DL2/data/original')
    
    for file in files:
        # �F���Ώۃt�@�C���̓ǂݍ���
        image_path = "C:/DL2/data/original/" + file
        image = cv2.imread(image_path)
        
        if isinstance(image,type(None)):
            continue
        
        # �摜�̃T�C�Y�A�`�����l�������擾
        image_height,image_width, image_channel = image.shape[:3]
        
        # ���m�N���摜�͉��H���Ȃ�
        if (image_channel!=3):
            continue
        
        # ���m��̓���m�F�p
        display = cv2.imread(image_path)

        # �O���[�X�P�[���ɕϊ�
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ��F���p�����ʂ̃t�@�C���w��
        cascade_path = "C:/DL2/haarcascade_frontalface_alt.xml"
        
        # �J�X�P�[�h���ފ�̗p��
        #�@��
        cascade = cv2.CascadeClassifier(cascade_path)
        #�@��
        eye_cascade_path="C:/DL2/haarcascade_eye.xml"
        eye_cascade=cv2.CascadeClassifier(eye_cascade_path)

        # ��F���̎��s
        minFaceSize= (int(image_height/8),int(image_width/5)) # ��ʂɑ΂��Ċ�͂���Ȃ�ɑ傫�� �f�[�^�Z�b�g�ˑ�
        facerecog = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=minFaceSize)
        
        if len(facerecog) == 1: # �炪1�m�F���ꂽ�Ƃ��̂ݏ���
            # �㏑������摜�̓ǂݍ���
            ol_imgae_path = random_stamp()    
            ol_image = cv2.imread(ol_imgae_path,cv2.IMREAD_UNCHANGED)   # �A���t�@�`�����l��(����)���ǂ݂��ނ悤��IMREAD_INCHANGED���w��
 
            # �F��������ɑ΂��ĉ��H������
            for rect in facerecog:
                # �F�����ʂ̕`��
                rectangle_func(display,rect,(0,0,255))
   
                # �F���͈͂ɂ��킹�ĉ摜�����T�C�Y
                resized_ol_image = resize_image(ol_image, rect[2], rect[3])                
                
                # �F��������͈͓̔��Ŗڂ�F������
                minEyeSize =(int(rect[2]/20),int(rect[3]/50)) #��ɔ�ׂĂ���Ȃɖڂ͏������Ȃ�
                maxEyeSize =(int(rect[2]/2),int(rect[3]/2)) #��ɔ�ׂĂ���Ȃɖڂ͑傫���Ȃ�
                roi_gray = image_gray[rect[1]:int(rect[1]+0.7*rect[3]), rect[0]:rect[0]+rect[2]] # �ڂ͊�̏�̕��ɑ��݂���
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=1, minSize=minEyeSize, maxSize=maxEyeSize)
                if (len(eyes) == 2 ): # �ڂ�2�m�F���ꂽ�Ƃ��̂ݖڊg��
                    print(eyes.shape)
                    for rect_eye in eyes:
                        # �ڂ̌��摜���ł̈ʒu
                        rect_eye_sum = [0,0,0,0]
                        rect_eye_sum[0]=rect_eye[0]+rect[0]
                        rect_eye_sum[1]=rect_eye[1]+rect[1]
                        rect_eye_sum[2]=rect_eye[2]
                        rect_eye_sum[3]=rect_eye[3]
                        # �ڂ̌��m���ꂽ�ꏊ��`��
                        rectangle_func(display,rect_eye_sum,(255,0,0))
                        #�@�ڂ��g�傷��͈͂����肷��
                        ## OpenCV�̖ڌ��o��ł͓����ӂ̔��т�����܂ł�ڂ̋�`�Ƃ��ďo�͂��邽�߁A�ꕔ�̂݊g�傷��
                        rect_eye_expand = [0,0,0,0]
                        rect_eye_expand[0]=rect_eye[0]+rect[0]
                        rect_eye_expand[1]=rect_eye[1]+int(0.3*rect_eye[3])+rect[1]
                        rect_eye_expand[2]=rect_eye[2]
                        rect_eye_expand[3]=int(0.6*rect_eye[3])
                        #�@�ڂ��g�傷��
                        deform_part(image,rect_eye_expand,1.2)
                        '''�����ڂ̊g��ʂ������_���ɂ�������΂ǂ����i�X�^���v�ɂ���Ċg��ʂ͌��܂��Ă���̂�
                        ���̎��_�Ń����_���ɂ����肩�̓X�^���v�����߂����Ɋg��ʂ����߂��ق����ǂ��j
                        if flag_eye == 1:
                            deform_part(image,rect_eye_expand,1.2)
                        elif flag_eye == 2:
                            deform_part(image,rect_eye_expand,1.25)
                        else:
                            deform_part(image,rect_eye_expand,1.3)
                        # �ڂ̉������܂���
                        rect_eye_down = [0,0,0,0]
                        rect_eye_down[0] = rect_eye[0]+rect[0]-5
                        rect_eye_down[1] = rect_eye[1]+rect_eye[3]+rect[1]-5
                        rect_eye_down[2] = rect_eye[2]+10
                        rect_eye_down[3] = 10
                        part_of_blur(image,rect_eye_down)
                        '''
                """        
                # �{���k������ -> �ȉ��̎����͖����������B��̌��o�������t�@�W�[�Ȃ̂�
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
                # �X�^���v��K�p����
                image = overlayOnPart(image, resized_ol_image, rect[0], rect[1])

            #�@�摜�̖��邳��ς���
            # �K���}�萔�̒�`
            gamma = 1.2
            # ���b�N�A�b�v�e�[�u���̐���
            look_up_table = np.ones((256, 1), dtype = 'uint8' ) * 0
            for i in range(256):
                look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
            # �K���}�ϊ���̏o��
            image = cv2.LUT(image, look_up_table)

            # �F�����ʂ̏o��
            cv2.imwrite("C:/DL2/data/snow/" + file , image)
            
            # �F�����ʂ̓���m�F�o��
            cv2.imwrite("C:/DL2/data/display/"+file,display)

# �摜�̈ꕔ���w�肳�ꂽ�g�嗦�Ńf�t�H��������
def deform_part(_image_color,_roi_deform, _magY,_top = 1): # �Ώۂ̃J���[�摜�A�f�t�H��������͈́A�c�̊g�嗦�A�㑵������������    
    # �摜�̈ꕔ���g�債���摜������
    deform_color = np.copy(_image_color[_roi_deform[1]:_roi_deform[1]+_roi_deform[3], _roi_deform[0]:_roi_deform[0]+_roi_deform[2], :])
    deform_color = cv2.resize(deform_color,(_roi_deform[2],int(_magY*_roi_deform[3])))
    if _top==1:
        _image_color[_roi_deform[1]:_roi_deform[1]+int(_magY*_roi_deform[3]), _roi_deform[0]:_roi_deform[0]+_roi_deform[2], :]=deform_color
    else:
        _image_color[_roi_deform[1]+_roi_deform[3]-int(_magY*_roi_deform[3]):_roi_deform[1]+_roi_deform[3], _roi_deform[0]:_roi_deform[0]+_roi_deform[2], :]=deform_color
        
# �摜�̈ꕔ�𕽊�������
        
# png�̉摜��\�����
def overlayOnPart(src_image, overlay_image, posX, posY):

    # �I�[�o���C�摜�̃T�C�Y���擾
    ol_height, ol_width = overlay_image.shape[:2]

    # OpenCV�̉摜�f�[�^��PIL�ɕϊ�
    #�@BGRA����RGBA�֕ϊ�
    src_image_RGBA = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    overlay_image_RGBA = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2RGBA)

    #�@PIL�ɕϊ�
    src_image_PIL=Image.fromarray(src_image_RGBA)
    overlay_image_PIL=Image.fromarray(overlay_image_RGBA)

    # �����̂��߁ARGBA���[�h�ɕύX
    src_image_PIL = src_image_PIL.convert('RGBA')
    overlay_image_PIL = overlay_image_PIL.convert('RGBA')

    # �����傫���̓��߃L�����p�X��p��
    tmp = Image.new('RGBA', src_image_PIL.size, (255, 255,255, 0))
    # �p�ӂ����L�����p�X�ɏ㏑��
    tmp.paste(overlay_image_PIL, (posX, posY), overlay_image_PIL)
    # �I���W�i���ƃL�����p�X���������ĕۑ�
    result = Image.alpha_composite(src_image_PIL, tmp)
    result_np=np.asarray(result)
    return  cv2.cvtColor(result_np[:,:,0:3], cv2.COLOR_RGB2BGR)
       
def resize_image(_image, _width,_height): # ���łɂ̓o�O����B

    # ���X�̃T�C�Y���擾
    org_height, org_width = _image.shape[:2]

    # �傫�����̃T�C�Y�ɍ��킹�ďk��
    if float(_height)/org_height > float(_width)/org_width:
        ratio = float(_height)/org_height
    else:
        ratio = float(_width)/org_width

    # ���T�C�Y
    resized = cv2.resize(_image,(int(org_width*ratio), int(org_height*ratio)))

    return resized   
        
if __name__ == '__main__':
    stamp_on_face()