wget "https://drive.google.com/uc?export=download&id=1X85yc_7Wdo928LOGa8u2xdBpx3zUYhKj" -O snow_japanese_train.zip
wget "https://drive.google.com/uc?export=download&id=1vm66H6I_CrLboV2msXcR61grUMOPvGyI" -O snow_japanese_talent_valid.zip
curl -c /tmp/cookie.txt -s -L "https://drive.google.com/uc?export=download&id=14gIzFuRbhVkmDrbgjYIWsfmwiCBTqwn_" |grep confirm |  sed -e "s/^.*confirm=\(.*\)&amp;id=.*$/\1/" | xargs -I{} \
curl -b /tmp/cookie.txt  -L -o snow_japanese_talent_train.zip "https://drive.google.com/uc?confirm={}&export=download&id=14gIzFuRbhVkmDrbgjYIWsfmwiCBTqwn_"
wget "https://drive.google.com/uc?export=download&id=1ZsYPbbsQ_ypwi-b0O13_EihDGe6Y_nBv" -O original_japanese_valid.zip
curl -c /tmp/cookie.txt -s -L "https://drive.google.com/uc?export=download&id=1aqhOk1T5iRRP2qlMjkFvBBleW4_3N7Uj" |grep confirm |  sed -e "s/^.*confirm=\(.*\)&amp;id=.*$/\1/" | xargs -I{} \
curl -b /tmp/cookie.txt  -L -o original_japanese_train.zip "https://drive.google.com/uc?confirm={}&export=download&id=1aqhOk1T5iRRP2qlMjkFvBBleW4_3N7Uj"
wget "https://drive.google.com/uc?export=download&id=1nTPG-xxN7494TaqRzxVpRLuSyoJ21lLu" -O original_japanese_talent_valid.zip
curl -c /tmp/cookie.txt -s -L "https://drive.google.com/uc?export=download&id=1RYE-EORYAnNXocinXoIBVCVwXSCRDiak" |grep confirm |  sed -e "s/^.*confirm=\(.*\)&amp;id=.*$/\1/" | xargs -I{} \
curl -b /tmp/cookie.txt  -L -o original_japanese_talent_train.zip "https://drive.google.com/uc?confirm={}&export=download&id=1RYE-EORYAnNXocinXoIBVCVwXSCRDiak"
wget "https://drive.google.com/uc?export=download&id=11nDfQXU1iu0j0LHDCzNc3wBSuZdlows6" -O snow_japanese_valid.zip

# unzip
unzip snow_japanese_train.zip
unzip snow_japanese_talent_valid.zip
unzip snow_japanese_talent_train.zip
unzip snow_japanese_valid.zip
unzip original_japanese_valid.zip
unzip original_japanese_train.zip
unzip original_japanese_talent_valid.zip
unzip original_japanese_talent_train.zip

# delete zip files
rm *.zip
