from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session


import numpy as np
import os
import tensorflow as tf
import cv2

from PIL import Image, ImageDraw
from google.protobuf import text_format
from werkzeug.utils import secure_filename
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential, load_model



from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils.visualization_utils import visualize_boxes_and_labels_on_image_array

### Model preparation variable

NUM_CLASSES = 1
PATH_TO_CKPT = 'frozen_inference_graph.pb'
PATH_TO_LABELS = 'object-detection.pbtxt'

def graph(PATH_TO_CKPT):
    with tf.gfile.GFile(PATH_TO_CKPT, "rb") as f:
        graph_def = tf.GraphDef()
        proto_b = f.read()
        text_format.Merge(proto_b, graph_def) 
        graph_def.ParseFromString(text_format)

    with tf.Graph().as_default() as detection_graph:
        tf.import_graph_def(graph_def, name="")
    return detection_graph


label_map = label_map_util.load_labelmap('object-detection.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                    real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                    real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, use_normalized_coordinates=True):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    return (left, right, top, bottom)
# Match contours to license plate or character template
def find_contours(dimensions, img):
    # Find all contours in the image
    image = img.copy()
    cntrs, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]


    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        cv2.rectangle(image, (intX, intY), (intX + intWidth, intY + intHeight), (0, 0, 255), 2)

        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
            x_cntr_list.append(intX)  # stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((28, 28))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (24, 24))

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:26, 2:26] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[26:28, :] = 0
            char_copy[:, 26:28] = 0

            img_res.append(char_copy)  # List that stores the character's binary image (unsorted)
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)

    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])  # stores character images according to their index
    img_res = np.array(img_res_copy)
    cv2.imwrite('segmentation.jpg', image)

    return img_res


def segment_characters(image) :
    image = cv2.imread(image)
    # Preprocess cropped license plate image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_erode = cv2.erode(img_binary, (3,3))

    LP_WIDTH = img_erode.shape[0]
    LP_HEIGHT = img_erode.shape[1]


    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2*LP_HEIGHT/3]

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_erode)
    return char_list

def predict_on_segmentation(model,image_path):
    license_plate = []
    license_plate_number = ''
    model = load_model(model)
    # Detect chars
    digits = segment_characters(image_path)


    for d in digits:
        d = np.reshape(d, (1,28,28,1))
        out = model.predict(d)
        # Get max pre arg
        p = []
        precision = 0
        for i in range(len(out)):
            z = np.zeros(36)
            z[np.argmax(out[i])] = 1.
            precision = max(out[i])
            p.append(z)
        prediction = np.array(p)

        # Inverse one hot encoding
        alphabets = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        classes = []
        for a in alphabets:
            classes.append([a])
        ohe = OneHotEncoder()
        ohe.fit(classes)
        pred = ohe.inverse_transform(prediction)
        license_plate.append(pred[0][0])

        if precision > 0:
            print('Prediction : ' + str(pred[0][0]) + ' , Precision : ' + str(precision))

    license_plate_number = ''.join(license_plate)
    print("License Number :" + license_plate_number )
    return license_plate_number

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # used to clear chache

@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')


@app.route("/description")
def description():
    return render_template('description.html')


@app.route('/img/<path:path>')
def send_img(path):
    return send_from_directory('./', path)

@app.route('/detect_plate', methods=['POST', 'GET'])
def detectPlate():
    global PATH_TO_CKPT, category_index
    detection_graph = graph(PATH_TO_CKPT)
    if request.method == 'POST':
        file = request.files['image']
        if not file: return render_template('index.html', label="No file")

        original_image = file.filename
        print(original_image)
        filename = secure_filename(file.filename)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:
                image = Image.open(file)
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                output_dict = run_inference_for_single_image(image_np, detection_graph)
                for boxes, score in zip(output_dict['detection_boxes'], output_dict['detection_scores']):
                    if score > 0.40:
                        ymin, xmin, ymax, xmax = boxes
                        left, right, top, bottom = draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax)
                        x = int(left)
                        x1 = int(right)
                        y = int(top)
                        y1 = int(bottom)
                        crop = image_np[y:y1, x:x1]
                        cv2.imwrite('number_plate.jpg', crop)

                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                im = Image.fromarray(image_np)
                im.save('Detected_plate.jpg')
                filename = 'Detected_plate.jpg'

    return render_template('segment.html', label=filename)

char_list = []
@app.route('/segment_characters', methods=['POST', 'GET'])
def segmentation():
    if request.method == 'POST':
        segment_characters('number_plate.jpg')
        filename = 'segmentation.jpg'
        return render_template('predict.html', label= filename)


license_plate = []
@app.route('/predict_characters', methods=['POST', 'GET'])

def make_prediction():
    if request.method == 'POST':
        predicted_text = predict_on_segmentation(model= 'cnn_classifier.h5', image_path='number_plate.jpg')
        filename = 'number_plate.jpg'
        return render_template('result.html', image=filename, label= str(predicted_text))


if __name__ == '__main__':
    app.run(debug=True)
