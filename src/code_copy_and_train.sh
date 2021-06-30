#!/bin/bash
# 여기있는 파일  모두 타겟 폴더로 복사한다.
# 그리고 train.py 실행

target_dir='../train_logs/210625_vnet2'
target_src_dir=$target_dir'/src'
echo $target_dir

mkdir -p $target_dir
mkdir -p $target_src_dir
cp -r * $target_src_dir

cd $target_src_dir
python train.py


# !!실행전 확인 사항!!
# target_dir 확인
# train.py에서 gpu 지정 확인
# 