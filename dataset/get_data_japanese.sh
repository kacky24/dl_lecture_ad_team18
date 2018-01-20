curl -c /tmp/cookie.txt -s -L "https://drive.google.com/uc?export=download&id=1N13O1xdqDY6eUUpuNpFm8VH8M1-gzyFQ" |grep confirm |  sed -e "s/^.*confirm=\(.*\)&amp;id=.*$/\1/" | xargs -I{} \
curl -b /tmp/cookie.txt -L -o snow_japanese_train.zip "https://drive.google.com/uc?confirm={}&export=download&id=1N13O1xdqDY6eUUpuNpFm8VH8M1-gzyFQ" 
wget "https://drive.google.com/uc?export=download&id=1Y5TqsQFQJLPbgG9-wMUH4f5RmxF3Ts1f" -O snow_japanese_valid.zip
curl -c /tmp/cookie.txt -s -L "https://drive.google.com/uc?export=download&id=1ZP31-pai5f5DaFEYb4zTakMiUtTv3OOV" |grep confirm |  sed -e "s/^.*confirm=\(.*\)&amp;id=.*$/\1/" | xargs -I{} \
curl -b /tmp/cookie.txt -L -o original_japanese_train.zip "https://drive.google.com/uc?confirm={}&export=download&id=1ZP31-pai5f5DaFEYb4zTakMiUtTv3OOV" 
wget "https://drive.google.com/uc?export=download&id=1REPImTW8_2B97bFcpTD0f-ult58ZEYMH" -O original_japanese_valid.zip

wget "https://drive.google.com/uc?export=download&id=1vm66H6I_CrLboV2msXcR61grUMOPvGyI" -O snow_japanese_talent_valid.zip
curl -c /tmp/cookie.txt -s -L "https://drive.google.com/uc?export=download&id=14gIzFuRbhVkmDrbgjYIWsfmwiCBTqwn_" |grep confirm |  sed -e "s/^.*confirm=\(.*\)&amp;id=.*$/\1/" | xargs -I{} \
curl -b /tmp/cookie.txt  -L -o snow_japanese_talent_train.zip "https://drive.google.com/uc?confirm={}&export=download&id=14gIzFuRbhVkmDrbgjYIWsfmwiCBTqwn_"
wget "https://drive.google.com/uc?export=download&id=1nTPG-xxN7494TaqRzxVpRLuSyoJ21lLu" -O original_japanese_talent_valid.zip
curl -c /tmp/cookie.txt -s -L "https://drive.google.com/uc?export=download&id=1RYE-EORYAnNXocinXoIBVCVwXSCRDiak" |grep confirm |  sed -e "s/^.*confirm=\(.*\)&amp;id=.*$/\1/" | xargs -I{} \
curl -b /tmp/cookie.txt  -L -o original_japanese_talent_train.zip "https://drive.google.com/uc?confirm={}&export=download&id=1RYE-EORYAnNXocinXoIBVCVwXSCRDiak"

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
