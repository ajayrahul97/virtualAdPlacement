import argparse
import time
from pathlib import Path
import imutils
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import sys

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
	"""
	@brief      Overlays a transparant PNG onto another image using CV2
	
	@param      background_img    The background image
	@param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
	@param      x                 x location to place the top-left corner of our overlay
	@param      y                 y location to place the top-left corner of our overlay
	@param      overlay_size      The size to scale our overlay to (tuple), no scaling if None
	
	@return     Background image with overlay on top
	"""
	
	bg_img = background_img.copy()
	
	if overlay_size is not None:
		img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

	# Extract the alpha mask of the RGBA image, convert to RGB 
	b,g,r,a = cv2.split(img_to_overlay_t)
	overlay_color = cv2.merge((b,g,r))
	
	# Apply some simple filtering to remove edge noise
	mask = cv2.medianBlur(a,3)

	h, w, _ = overlay_color.shape
	roi = bg_img[y:y+h, x:x+w]

	# Black-out the area behind the logo in our original ROI
	img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))
	
	# Mask out the logo from the logo image.
	img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

	# Update the original image with our new ROI
	bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

	return bg_img

def find_fground_box(x, img):
    # Plots one bounding box on image img
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    print(c1, c2)

    ad_frame = cv2.imread("/Users/aashish/Desktop/HackPSU/samples/ads/mt-bank-logo.png", -1)
    # print("dsdsd: ", c1[0], img.shape[1] - c2[0], c2[0])
    

    if(c1[0] > c1[1] and c1[0] > img.shape[1] - c2[0]):
        ad_frame = imutils.resize(ad_frame, width=((c1[1])*2)//3)
        # ad_frame = imutils.resize(ad_frame, height=(c1[1]*6)//3)

        ad_h, ad_w = ad_frame.shape[:2]
        y_ad_center = c1[1] + (c2[1] - c1[1])//2
        x_ad_center = c1[0]//2 
        ad_left = x_ad_center - ad_w//2
        ad_right = x_ad_center + (ad_w - ad_w//2)
        ad_top = y_ad_center - ad_h//2
        ad_bottom = y_ad_center + (ad_h - ad_h//2)
        # print(ad_right - ad_left, ad_bottom - ad_top, ad_w, ad_h)
        
    elif(img.shape[1] - c2[0] > c1[1] and img.shape[1] - c2[0] > c1[0]):
        ad_frame = imutils.resize(ad_frame, width=((img.shape[1] - c2[1])*2)//3)
        # ad_frame = imutils.resize(ad_frame, height=(c1[1]*6)//3)

        ad_h, ad_w = ad_frame.shape[:2]
        y_ad_center = c1[1] + (c2[1] - c1[1])//2
        x_ad_center = (img.shape[0] - c2[0])//2  
        ad_left = x_ad_center - ad_w//2
        ad_right = x_ad_center + (ad_w - ad_w//2)
        ad_top = y_ad_center - ad_h//2
        ad_bottom = y_ad_center + (ad_h - ad_h//2)
        # print(ad_right - ad_left, ad_bottom - ad_top, ad_w, ad_h)
    
    else:
        ad_frame = imutils.resize(ad_frame, width=((c2[1] - c1[1])*1)//2)
        # ad_frame = imutils.resize(ad_frame, height=(c1[0]*1)//3)

        ad_h, ad_w = ad_frame.shape[:2]
        y_ad_center = c1[1]//2
        x_ad_center = c1[0] + (c2[0] - c1[0])//2
        ad_left = x_ad_center - ad_w//2
        ad_right = x_ad_center + (ad_w - ad_w//2)
        ad_top = y_ad_center - ad_h//2
        ad_bottom = y_ad_center + (ad_h - ad_h//2)
        print(ad_right - ad_left, ad_bottom - ad_top, ad_w, ad_h)
    
    
    img = overlay_transparent(img, ad_frame, ad_left, ad_top)

    # img = Image.fromarray(img)
    # ad_frame = Image.fromarray(ad_frame)

    # img = img.convert("RGBA")
    # ad_frame = ad_frame.convert("RGBA")

    # img.paste(ad_frame, (ad_left, ad_top), ad_frame)

    # img.save("./runs/detect/test.png", format="png")

    # cv2.imwrite('test.png', )
    # sys.exit(0)

    #coords_ad_img = np.array([[ad_left, ad_top], [ad_left, ad_bottom], [ad_right, ad_bottom],[ad_right, ad_top]])
    # print(img[ad_left:ad_right, ad_top:ad_bottom].shape, ad_frame.shape)
    # img[ad_top:ad_bottom, ad_left:ad_right] = ad_frame[0:ad_bottom - ad_top, 0:ad_right-ad_left]
    return(img)

    #superimpose_img = np.array([[0,0],[0, ad_h-1], [ad_w-1, ad_h-1],[ad_w-1, 0]])

    #pt, status = cv2.findHomography(coords_ad_img, superimpose_img)
    #return(cv2.warpPerspective(ad_frame, pt, (img.shape[1], img.shape[0])))

    #coords_ad_img = np.array([[ad_left, ad_top], [ad_left, ad_bottom], [ad_right, ad_bottom],[ad_right, ad_top]])
 

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    frame_count = 0
    foreground_coords = None
    for path, img, im0s, vid_cap in dataset:
        frame_count += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                # print(im0.shape, det[:, :4])
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                min_coords = [np.inf, np.inf]
                max_coords = [0,0]
                for *xyxy, conf, cls in reversed(det):
                    # print(f'{names[int(cls)]} {conf:.2f}')
                    if(f'{names[int(cls)]} {conf:.2f}'.startswith('person')):
                        min_coords[0] = min(min_coords[0], int(xyxy[0]))
                        min_coords[1] = min(min_coords[1], int(xyxy[1]))
                        max_coords[0] = max(max_coords[0], int(xyxy[2]))
                        max_coords[1] = max(max_coords[1], int(xyxy[3]))

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        # 
                
                if(foreground_coords == None):
                    foreground_coords = min_coords + max_coords
                    
                if(frame_count == 30):
                    frame_count = 0
                    center_new_frame_x = abs(min_coords[0] + (max_coords[0]-min_coords[0])//2)
                    center_new_frame_y = abs(min_coords[1] + (max_coords[1]-min_coords[1])//2)
                    existing_fg_x = abs(foreground_coords[0] + (foreground_coords[2]-foreground_coords[0])//2)
                    existing_fg_y = abs(foreground_coords[1] + (foreground_coords[3]-foreground_coords[1])//2) 
                                             
                    if(center_new_frame_x - existing_fg_x > 30 and center_new_frame_y - existing_fg_y > 30):
                        im0 = find_fground_box(min_coords + max_coords, im0)
                        foreground_coords = min_coords + max_coords 

                if(min_coords != [np.inf, np.inf]):
                    plot_one_box(min_coords + max_coords, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    im0 = find_fground_box(foreground_coords, im0) 
                    
                
                

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
