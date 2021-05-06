import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import cv2
import requests
import simplejson
import time
import base64
from test import test_result

app = Flask(__name__)
app.config['DEBUG'] = False


def tojson(result):
    return {
        'result': str(result)
    }


@app.route('/')
def hello_world():
    return '测试成功'


@app.route('/test/', methods=['post'])
def test():
    if not request.data:  # 检测是否有数据
        return 'fail'
    student = request.data.decode('utf-8')
    # 获取到POST过来的数据，因为我这里传过来的数据需要转换一下编码。根据晶具体情况而定
    student_json = simplejson.loads(student)
    # 把区获取到的数据转为JSON格式。
    img_str = student_json['recognize_img']

    # print(student_json['recognize_img'])
    # img_str = img_str_['recognize_img']
    img_decode_ = img_str.encode('ascii')  # ascii编码
    img_decode = base64.b64decode(img_decode_)  # base64解码
    img_np = np.frombuffer(img_decode, np.uint8)  # 从byte数据读取为np.array形式
    img = cv2.imdecode(img_np, cv2.COLOR_RGB2BGR)  # 转为OpenCV形式
    a = time.time()
    # 显示图像
    path = './using/' + str(a) + '.jpg'
    cv2.imwrite(path, img)
    # print(path)
    data = []
    result = test_result(path)
    # print(result)
    # print(tojson(str(result)))
    data.append(tojson(str(result)))
    # print(data)
    # print(type(simplejson.dumps(data)))
    return simplejson.dumps(data, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3270)
