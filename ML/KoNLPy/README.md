# KoNLPy 사용을 위한 환경설정

## KoNLPy 설치 - windows 10 (python=3.7)

JDK 1.7+ 설치
JAVA_HOME 설정

1. create new evn
```
conda create -n konlp python=3.7 anaconda
conda info --envs
conda activate konlp

pip install tensorflow-gpu  (v2.3)
```

2. install konlp
```
pip install konlpy (JPype1 함께 설치됨)
```

3. Mecab 설치 (optional)
```
https://cleancode-ws.tistory.com/97
whl 파일은 env 폴더에 복사
```
