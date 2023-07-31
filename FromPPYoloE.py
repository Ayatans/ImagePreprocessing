def process_image(input_image, size):
    """输入图片与处理方法，按照PP-Yoloe模型要求预处理图片数据
    Args:
        input_image (uint8): 输入图片矩阵
        size (int): 模型输入大小
    Returns:
        float32: 返回处理后的图片矩阵数据
    """
    max_len = max(input_image.shape)
    img = np.zeros([max_len, max_len, 3], np.uint8)
    img[0:input_image.shape[0], 0:input_image.shape[1]] = input_image  # 将图片放到正方形背景中
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # BGR转RGB
    img = cv.resize(img, (size, size), cv.INTER_NEAREST)  # 缩放图片
    img = np.transpose(img, [2, 0, 1])  # 转换格式
    img = img / 255.0  # 归一化
    img = np.expand_dims(img, 0)  # 增加维度
    return img


def process_result(box_results, conf_results):
    """按照PP-Yolove模型输出要求，处理数据，非极大值抑制，提取预测结果
    Args:
        box_results (float32): 预测框预测结果
        conf_results (float32): 置信度预测结果
    Returns:
        float: 预测框
        float: 分数
        int: 类别
    """
    conf_results = np.transpose(conf_results, [0, 2, 1])  # 转换数据通道
    # 设置输出形状
    box_results = box_results.reshape(8400, 4)
    conf_results = conf_results.reshape(8400, 80)
    scores = []
    classes = []
    boxes = []
    for i in range(8400):
        conf = conf_results[i, :]  # 预测分数
        score = np.max(conf)  # 获取类别
        # 筛选较小的预测类别
        if score > 0.5:
            classes.append(np.argmax(conf))
            scores.append(score)
            boxes.append(box_results[i, :])
    scores = np.array(scores)
    boxes = np.array(boxes)
    # 非极大值抑制筛选重复的预测结果
    picked_boxes, picked_score, indexs = nms(boxes, scores)
    print(indexs)
    # 处理非极大值抑制后的结果
    result_box = []
    result_score = []
    result_class = []
    for i, index in enumerate(indexs):
        result_score.append(scores[index])
        result_box.append(boxes[index, :])
        result_class.append(classes[index])
    # 返沪结果转为矩阵
    return np.array(result_box), np.array(result_score), np.array(result_class)


def draw_box(image, boxes, scores, classes, lables):
    """将预测结果绘制到图像上

    Args:
        image (uint8): 原图片
        boxes (float32): 预测框
        scores (float32): 分数
        classes (int): 类别
        lables (str): 标签

    Returns:
        uint8: 标注好的图片
    """
    scale = max(image.shape) / 640.0  # 缩放比例
    for i in range(len(classes)):
        box = boxes[i, :]

        x1 = int(box[0] * scale)
        y1 = int(box[1] * scale)
        x2 = int(box[2] * scale)
        y2 = int(box[3] * scale)

        lable = lables[classes[i]]
        score = scores[i]
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv.LINE_8)
        cv.putText(image, lable + ":" + str(score), (x1, y1 - 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return image


def read_lable(lable_path):
    """读取lable文件

    Args:
        lable_path (str): 文件路径

    Returns:
        str: _description_
    """
    f = open(lable_path)
    lable = []
    line = f.readline()
    while line:
        lable.append(line)
        line = f.readline()
    f.close()
    return lable


def nms(bounding_boxes, confidence_score):
    '''
    :param bounding_boxes: 候选框列表，[左上角坐标, 右下角坐标], [min_x, min_y, max_x, max_y], 原点在图像左上角
    :param confidence_score: 候选框置信度
    :param threshold: IOU阈值
    :return: 抑制后的bbox和置信度
    '''
    picked = []

    for i in range(confidence_score.shape[-1]):
        if confidence_score[i] > 0.35:
            picked.append(i)
    bounding_boxes = bounding_boxes[picked, :]
    confidence_score = confidence_score[picked]
    # 如果没有bbox，则返回空列表
    if len(bounding_boxes) == 0:
        return [], []

    # bbox转为numpy格式方便计算
    boxes = np.array(bounding_boxes)

    # 分别取出bbox的坐标
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # 置信度转为numpy格式方便计算
    score = np.array(confidence_score)  # [0.9  0.75 0.8  0.85]

    # 筛选后的bbox和置信度
    picked_boxes = []
    picked_score = []

    # 计算每一个框的面积
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # 将score中的元素从小到大排列，提取其对应的index(索引)，然后输出到order
    order = np.argsort(score)  # [1 2 3 0]
    indexs = []
    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        # 取出最大置信度的索引
        index = order[-1]
        indexs.append(index)
        # Pick the bounding box with largest confidence score
        # 将最大置信度和最大置信度对应的框添加进筛选列表里
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # 求置信度最大的框与其他所有框相交的长宽，为下面计算相交面积做准备
        # 令左上角为原点，
        # 两个框的左上角坐标x取大值，右下角坐标x取小值，小值-大值+1==相交区域的长度
        # 两个框的左上角坐标y取大值，右下角坐标y取小值，小值-大值+1==相交区域的高度
        # 这里可以在草稿纸上画个图，清晰明了
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # 计算相交面积，当两个框不相交时，w和h必有一个为0，面积也为0
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # 计算IOU
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        # 保留小于阈值的框的索引
        left = np.where(ratio < 0.25)
        # 根据该索引修正order中的索引（order里放的是按置信度从小到大排列的索引）
        order = order[left]

    return picked_boxes, picked_score, indexs


class Predictor:
    """
    OpenVINO 模型推理器
    """

    def __init__(self, model_path):
        ie_core = Core()
        model = ie_core.read_model(model=model_path)
        self.compiled_model = ie_core.compile_model(model=model, device_name="CPU")

    def get_inputs_name(self, num):
        return self.compiled_model.input(num)

    def get_outputs_name(self, num):
        return self.compiled_model.output(num)

    def predict(self, input_data):
        return self.compiled_model([input_data])


class Predictor:
    """
    OpenVINO 模型推理器
    """

    def __init__(self, model_path):
        ie_core = Core()
        model = ie_core.read_model(model=model_path)
        self.compiled_model = ie_core.compile_model(model=model, device_name="CPU")

    def get_inputs_name(self, num):
        return self.compiled_model.input(num)

    def get_outputs_name(self, num):
        return self.compiled_model.output(num)

    def predict(self, input_data):
        return self.compiled_model([input_data])


image_path = "/home/aistudio/image/demo_3.jpg"
image  =yoloe_infer(image_path)
image = image[:,:,::-1]     #把图像转成rgb
plt.imshow(image)
plt.show()
