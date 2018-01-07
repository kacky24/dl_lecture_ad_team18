#!/bin/bash

idcgan_dir=idcgan
pix2pix_dir=pix2pix

if [ -e $idcgan_dir ]; then
    :
else
    mkdir $idcgan_dir
fi

if [ -e $pix2pix_dir ]; then
    :
else
    mkdir $pix2pix_dir
fi


wget "https://www.dropbox.com/s/y058czhrjg76nih/096.ckpt" -O idcgan/096.ckpt
wget "https://www.dropbox.com/s/mlo1ay5ahyljvv5/200.ckpt" -O pix2pix/200.ckpt
