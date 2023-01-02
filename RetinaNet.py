from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
detector.loadModel()

detections, objects_path = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path , "objects.jpg"),
    output_image_path=os.path.join(execution_path , "new_objects.jpg"),
    minimum_percentage_probability=80,
    extract_detected_objects=True
)

for eachObject, eachObjectPath in zip(detections, objects_path):
    print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("Object's image saved in " + eachObjectPath)
    print("--------------------------------")
