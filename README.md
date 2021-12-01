# ner-demo
# 模型展示
![image](https://github.com/dreams-flying/ner-demo/blob/master/images/images.png)
# 所需环境
Python==3.6</br>
tensorflow==1.14.0</br>
keras==2.3.1</br>
bert4keras==0.10.6</br>
Flask==1.1.2</br>
# 项目目录
├── templates &emsp;&emsp;存放html文件</br>
├── images &emsp;&emsp;存放图片</br>
├── utils &emsp;&emsp;存放模型相关文件</br>
│ ├── bert4keras</br>
│ ├──  pretrained_model &emsp;&emsp;存放预训练模型</br>
│ ├──  save &emsp;&emsp;存放已训练好的模型</br>
│ ├── ner_predict.py &emsp;&emsp;ner预测及处理函数</br>
├── app.py&emsp;&emsp;Flask程序主入口</br>
# 使用说明
1.安装相关库</br>
2.切换到主目录，运行flask</br>
```
python app.py
```
3.打开浏览器，输入
```
localhost:5000/
```
# 数据说明
模型采用[人民日报](https://github.com/ThunderingII/nlp_ner/tree/master/data)数据进行训练，分三个类别：人名，地点，组织机构。
