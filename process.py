from ultralytics import YOLO
import json
import os

model = YOLO('yolov8x.pt')

def process_file(file_name):
    # Classes list
    '''
    names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    '''
    # Docs: https://docs.ultralytics.com/modes/predict/#image-and-video-formats
    results = model.track(
        source=f"input/{file_name}", 
        show=False, 
        save=False, 
        name=f"output/{file_name}", # Save file name
        persist=False, 
        classes=[2, 5, 7], # Classes to recognize
        conf=0.5, # Confidence Threshold
        vid_stride=2, # Skip some frames
    )

    data_points = []
    frame = 0
    for r in results:
        b = r.boxes
        for idx, coords in enumerate(b.xyxy):
            data_points.append({
                "file": file_name,
                "frame": frame,
                "id": int(b.id.numpy()[idx]) if b.id is not None else -1,
                "x": float(coords[0]),
                "y": float(coords[1]), # TODO: Get centroid instead of edge
                "x2": float(coords[2]),
                "y2": float(coords[3]),
                "class": float(b.cls[idx]),
                "conf": float(b.conf[idx])
            })
        frame = frame + 1

    json_object = json.dumps(data_points)
    with open(f"data/{file_name}.json", "w") as outfile:
        outfile.write(json_object)

for filename in os.listdir("input"):
    print(filename)
    process_file(filename)
    os.rename(f"input/{filename}", f"output/{filename}")