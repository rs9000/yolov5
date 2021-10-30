import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
import cv2
import yaml


def main():
    # Select device cuda:0 or cpu
    device = select_device('')
    # Load pretrained checkpoint
    weights = "yolov5s.pt"

    # Load model
    model = attempt_load(weights, map_location=device)
    print(model)

    # Define vars
    img_size = [640, 640]
    stride = 32
    input_image = "./data/images/bus.jpg"
    out_image = "./bus.jpg"

    # Read class names from ./data/coco.yaml
    stream = open('./data/coco.yaml', 'r')
    names = yaml.load(stream)['names']

    # Load images
    dataset = LoadImages(input_image, img_size=img_size, stride=stride, auto=True)

    # Iterate on image dataset
    for path, img, im0s, _ in dataset:
        # Convert image (img) from numpy to torch tensor
        img = torch.from_numpy(img).to(device)
        # Normalize image from interval [0-255] to the interval [0-1]
        img = img / 255.0
        # Add the batch dimension (3, h, w) to (1, 3, h, w)
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
            print("\nImage shape: ", img.shape)

        # Model prediction
        pred = model(img)[0]
        print("\nOutput model shape: ", pred.shape)
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)
        print("\nOutput NMS shape: ", pred[0].shape)

        s = "\n"
        annotator = Annotator(im0s, line_width=3, example=str(names))

        for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))

        print(f'{s}Done.')
        im0 = annotator.result()
        cv2.imwrite(out_image, im0)


if __name__ == '__main__':
    main()

