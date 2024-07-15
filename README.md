## How to setup
### 1. (optional) Set up a virtual environment
```bash
virtualenv .venv
source .venv/bin/activate
```
### 2. Install the required packages
```bash
pip install -r requirements.txt
```
### 3. Install Tesseract and enter path in .env
#### 1. Install Tessract
```bash
sudo apt-get isntall tesseract-ocr
```
#### 2. Set the tesseract path in the .env
### 4. Install llava
#### 1. Install ollama
https://ollama.com/download
#### 2. pull/run llava
```bash
ollama run llava
```
## How to run 
```bash
python3 src/main.py
```