import mxnet as mx
import numpy as np
import cv2

def unpack_rec_file(rec_file):
    record = mx.recordio.MXRecordIO(rec_file, 'r')
    header, _ = mx.recordio.unpack(record.read())
    print(header)

    for i in range(5):  # 只显示前5张图片
        item = record.read()
        header, img = mx.recordio.unpack_img(item)

        # 将 MXNet 图片数据转换为 OpenCV 格式
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        cv2.imwrite("C:\\Users\\yangc\\Developer\\Face-Recognition\\data\\images\\img_" + str(i) + ".jpg", img_cv2)

    cv2.destroyAllWindows()  # 关闭窗口

# 指定rec文件路径
rec_file = "C:\\Users\\yangc\\Developer\\data\\faces_umd\\train.rec"

# 调用解包函数
unpack_rec_file(rec_file)
