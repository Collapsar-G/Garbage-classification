import requests
import simplejson
import base64


# 首先将图片读入
# 由于要发送json，所以需要对byte进行str解码
def getByte(path):
    with open(path, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode('ascii')
    return img_str


if __name__ == '__main__':
    # path = './data/garbage_classify/train_data/img_2.jpg'
    path = './data/QQ图片20210306111130.jpg'
    img_str = getByte(path)
    # url = 'http://localhost:3270/test/'
    # url = 'https://collapsar.cn1.utools.club/test/'
    url = 'https://dachuang.collapsar.online/test/'

    data = {'recognize_img': img_str}
    json_mod = simplejson.dumps(data)
    # headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)
    # Chrome/81.0.4044.138 Safari/537.36'}
    res = requests.post(url=url, data=json_mod)
    print(res.text)
