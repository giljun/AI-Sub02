import pickle
from threading import Thread
import sqlite3

import numpy as np
from konlpy.tag import Okt
from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter
from scipy.sparse import lil_matrix


# slack 연동 정보 입력 부분
SLACK_TOKEN = "xoxb-732922739009-747806526096-Ky1D0CPXjquJmi9054KZoOLu"
SLACK_SIGNING_SECRET = "a4047071f49de1e04f8ab0a2de88447e"

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Req 2-2-1. pickle로 저장된 model.clf 파일 불러오기
with open('naive_model.clf', 'rb') as nai_model:
    naive_model = pickle.load(nai_model)
with open('rogistic_model.clf', 'rb') as rogi_model:
    rogistic_model = pickle.load(rogi_model)

pickle_obj = naive_model
word_indices = naive_model[1]
clf = naive_model[0]

# Req 2-2-2. 토큰화 및 one-hot 임베딩하는 전 처리
def preprocess():

    return None

# Req 2-2-3. 긍정 혹은 부정으로 분류
def classify():

    return None
    
# Req 2-2-4. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장
    

# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]


@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"

if __name__ == '__main__':
    app.run()
