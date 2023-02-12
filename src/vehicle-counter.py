# References:
# 1. Vehicle counter using python opencv and dlib -- https://github.com/jideilori/vehicle-counting
# 2. PyImageSearch: OpenCV Object Tracking -- https://pyimagesearch.com/2018/07/30/opencv-object-tracking/

import numpy as np
import cv2
import os
import shutil
import copy
from PIL import Image
import torch
from tqdm import tqdm

try:
    import gtimer as gt
except ModuleNotFoundError:
    %pip install gtimer
    import gtimer as gt

num_frames = 2_000
detection_frequency = 25
frame_skip_frequency = 5

threshold_score = 0.3 # Confidence threshold
lifetime_tracker = 20
border_width, border_height, width_inter_image = 20, 20, 10

def BGR_to_RGB(image): return image[:, :, ::-1]

# Check if 'point' is inside (or on) 'rectangle'.
def point_in_rectangle(point, rectangle):
    rect_point_1, rect_point_2 = rectangle[0], rectangle[1]
    
    return (
        (point[0] >= min(rect_point_1[0], rect_point_2[0])) and (point[0] <= max(rect_point_1[0], rect_point_2[0]))
    ) and (
        (point[1] >= min(rect_point_1[1], rect_point_2[1])) and (point[1] <= max(rect_point_1[1], rect_point_2[1]))
    )

# Check if 'rectangle_1' is completely contained in 'rectangle_2'.
def rectangle_contained_in_rectangle(rectangle_1, rectangle_2):
    return (
        point_in_rectangle(point = rectangle_1[0], rectangle = rectangle_2) 
        and
        point_in_rectangle(point = rectangle_1[1], rectangle = rectangle_2)
    )

def put_rectangle_opaque(frame, point_1, point_2, colour = (0, 255, 255), opacity = 0.7, thickness = 1):
    frame_temp = copy.deepcopy(frame)
    cv2.rectangle(
        img = frame_temp,
        pt1 = point_1,
        pt2 = point_2,
        color = (0, 255, 255),
        thickness = thickness
    )

    return cv2.addWeighted(
        src1 = frame, alpha = 1 - opacity, src2 = frame_temp, beta = opacity, gamma = 0,
    )

def put_text_opaque(
    frame, text, point, font = cv2.FONT_HERSHEY_SIMPLEX, font_size = 1, 
    text_colour = (255, 0, 0), opacity = 0.7, thickness = 1
):
    frame_temp = copy.deepcopy(frame)
    cv2.putText(
        img = frame_temp, 
        text = text, 
        org = point,
        fontFace = font, 
        fontScale = font_size, 
        color = text_colour, 
        thickness = thickness
    )

    return cv2.addWeighted(
        src1 = frame, alpha = 1 - opacity, src2 = frame_temp, beta = opacity, gamma = 0,
    )

def put_rectangle_with_title(
    frame, point_1, point_2, title, box_colour = (0, 255, 255), 
    opacity = 0.7, thickness = 1, title_colour = (0, 255, 255), font_size = 0.5, title_position = "nw"
):
    assert title_position in ["nw", "ne", "se", "sw"]
    frame = put_rectangle_opaque(
        frame = frame, point_1 = point_1, point_2 = point_2, colour = box_colour, 
        opacity = opacity, thickness = thickness
    )

    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width.
    (w, h), _ = cv2.getTextSize(text = title, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = font_size, thickness = 1)

    # frame = rectangle_with_opacity(
    #     frame = frame, point_1 = (point_1[0], point_1[1] - h), point_2 = (point_2[0], point_1[1]), 
    #     colour = box_colour, opacity = opacity, thickness = 1
    # )

    origin_map = {
        "nw": (point_1[0], point_1[1] - 5),
        "ne": (point_2[0] - w, point_1[1] - 5),
        "se": (point_2[0] - w, point_2[1] + h + 5),
        "sw": (point_1[0], point_2[1] + h + 5),
    }

    return put_text_opaque(
        frame = frame, text = title, point = origin_map.get(title_position),
        font = cv2.FONT_HERSHEY_SIMPLEX, font_size = font_size, 
        text_colour = title_colour, opacity = opacity, thickness = 1
    )

# Download video from YouTube

path_video = "/content/output"
file_video = "video.mp4"
# url_video = "https://youtu.be/MNn9qKG2UFI"
url_video = "https://youtu.be/nt3D26lrkho"

if (os.path.exists(path_video)): shutil.rmtree(path = path_video)
os.mkdir(path = path_video)

try:
    from pytube import YouTube
except ModuleNotFoundError:
    %pip install pytube
    from pytube import YouTube

_ = YouTube(url_video).streams.filter(res = "240p").first().download(
    output_path = path_video,
    filename = file_video
)

vid_cap = cv2.VideoCapture(os.path.join(path_video, file_video))
vid_cap.set(propId = cv2.CAP_PROP_FRAME_COUNT, value = num_frames)

# num_frames = int(vid_cap.get(propId = cv2.CAP_PROP_FRAME_COUNT))

frame_width, frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# rectangle_of_interest = ((0, 100), (frame_width, frame_height))
# rectangle_in_region = ((0, 430), (640, 480))
# rectangle_out_region = ((0, 0), (640, 50))

rectangle_of_interest = ((0, 100), (frame_width, frame_height))
rectangle_in_region = ((0, 200), (185, 240))
rectangle_out_region = ((220, 75), (285, 120))

tracker_type_dict = {
    "mosse": cv2.legacy.TrackerMOSSE,
    "csrt": cv2.TrackerCSRT,
    "kcf": cv2.TrackerKCF,
    "mf": cv2.legacy.TrackerMedianFlow,
    "tld": cv2.legacy.TrackerTLD
}
tracker_type = "mf"
tracker_name = tracker_type_dict[tracker_type]

model = torch.hub.load(
    repo_or_dir = "ultralytics/yolov5", model = "yolov5m",
    verbose = True, pretrained = True
)

class_id_list = [1, 2, 3, 5, 6, 7]
class_dict = model.names

is_active_in_counter, is_active_out_counter = True, True
frame_list_single, frame_list_side_by_side = [], []

def generate_output_frames(
    count_incoming = True, count_outgoing = True,
    set_regions_automatically = False, show_debug_info = False,
    show_video_debug_info = False, generate_video_side_by_side = False
):
    global frame_list_single, frame_list_side_by_side
    tracker_list, tracker_age_list = [], []
    in_count, out_count = 0, 0

    gt.reset()

    for index_cur_frame in tqdm(range(num_frames)):
        if (index_cur_frame % frame_skip_frequency == 0):
            # get frame from the video
            _, frame = vid_cap.read()
            gt.stamp(name = "Extract and process frame", unique = False)

            frame_copy = copy.deepcopy(frame)

            # Update the trackers.
            result_list = [tracker.update(frame_copy) for tracker in tracker_list]
            is_success_list = [result[0] for result in result_list]
            box_tracker_list = [result[1] for result in result_list]
            tracker_age_list = [age + 1 for age in tracker_age_list]

            for i, age in enumerate(tracker_age_list):
                box = box_tracker_list[i]
                tracker_center_x = box[0] + (box[2] // 2)
                tracker_center_y = box[1] + (box[3] // 2)
                if (
                    (age > lifetime_tracker)
                    and 
                    not point_in_rectangle(
                        point = (tracker_center_x, tracker_center_y), 
                        rectangle = 2 * np.array(rectangle_in_region)
                    )
                    and
                    not point_in_rectangle(
                        point = (tracker_center_x, tracker_center_y), 
                        rectangle = 2 * np.array(rectangle_out_region)
                    )
                ):
                    tracker_list.pop(i)
                    tracker_age_list.pop(i)
            
            gt.stamp(name = "Update trackers", unique = False)

            # Perform detection once every 'detection_frequency' iterations.
            if (index_cur_frame % (detection_frequency * frame_skip_frequency) == 0):
                patch = frame_copy[
                    rectangle_of_interest[0][1] : rectangle_of_interest[1][1],
                    rectangle_of_interest[0][0] : rectangle_of_interest[1][0]
                ]
                outputs = model.forward(patch)
                gt.stamp(name = "Generate detections", unique = False)

                score_list, bounding_box_list = [], []

                # Only keep the detections that have a high enough score and are inside 
                # the field of vision. Discard the rest.
                for detection in outputs.xywh[0].numpy():
                    score = detection[4]
                    class_id = detection[5]
                    if ((score > threshold_score) and (class_id in class_id_list)):
                        center_x = detection[0] + rectangle_of_interest[0][0]
                        center_y = detection[1] + rectangle_of_interest[0][1]
                        width = detection[2]
                        height = detection[3]
                        top_left_x = (center_x - (width / 2))
                        top_left_y = (center_y - (height / 2))
                        bottom_right_x = (center_x + (width / 2))
                        bottom_right_y = (center_y + (height / 2))
                        if (
                            # # The detection is inside the rectangle of interest
                            # rectangle_contained_in_rectangle(
                            #     rectangle_1 = ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)),
                            #     rectangle_2 = (rectangle_of_interest)
                            # )
                            # and
                            # The detection is not inside the in-region
                            not point_in_rectangle(
                                point = (bottom_right_x, bottom_right_y),
                                rectangle = rectangle_in_region
                            )
                            and 
                            # The detection is not inside the out-region
                            not point_in_rectangle(
                                point = (top_left_x, top_left_y), 
                                rectangle = rectangle_out_region
                            )
                        ):
                            score_list.append(score)
                            bounding_box_list.append(
                                [int(top_left_x), int(top_left_y), int(width), int(height)]
                            )
                gt.stamp(name = "Refine detections", unique = False)

                indices = np.arange(len(bounding_box_list))

                # Assign a new tracker to a new detection.
                for index in indices:
                    box_top_left_x, box_top_left_y, width, height = bounding_box_list[index]
                    # class_id, score = class_id_list[index], score_list[index]
                    box_bottom_right_x, box_bottom_right_y = box_top_left_x + width, box_top_left_y + height

                    # print("Bottom-left: ({}, {}), Top-right: ({}, {})".format(bottom_left_x, bottom_left_y, top_right_x, top_right_y))
                    
                    # Check if this "detection" is being tracked by an existing tracker.
                    tracking = False
                    for tracker, success, box in zip(tracker_list, is_success_list, box_tracker_list):
                        tracker_top_left_x, tracker_top_left_y = box[0], box[1]
                        tracker_bottom_right_x, tracker_bottom_right_y = tracker_top_left_x + box[2], tracker_top_left_y + box[3]

                        tracker_center_x = (tracker_top_left_x + tracker_bottom_right_x) // 2
                        tracker_center_y = (tracker_top_left_y + tracker_bottom_right_y) // 2
                        is_tracker_center_in_detection_box = point_in_rectangle(
                            point = (tracker_center_x, tracker_center_y),
                            rectangle = ((box_top_left_x, box_top_left_y), (box_bottom_right_x, box_bottom_right_y))
                        )
                        if (is_tracker_center_in_detection_box): tracking = True; break

                    # If no tracker is tracking this "detection", create a new tracker for this "detection".
                    if (not tracking):
                        tracker = tracker_name().create()
                        _ = tracker.init(
                            image = frame_copy,
                            boundingBox = (box_top_left_x, box_top_left_y, width, height)
                        )
                        tracker_list.append(tracker)
                        tracker_age_list.append(0)
                gt.stamp(name = "Check and add trackers", unique = False)

            for i, (tracker, box) in enumerate(zip(tracker_list, box_tracker_list)):                        
                tracker_top_left_x, tracker_top_left_y = int(box[0]), int(box[1])
                tracker_bottom_right_x, tracker_bottom_right_y = tracker_top_left_x + int(box[2]), tracker_top_left_y + int(box[3])
                
                cv2.rectangle(
                    img = frame_copy,
                    pt1 = (tracker_top_left_x, tracker_top_left_y),
                    pt2 = (tracker_bottom_right_x, tracker_bottom_right_y),
                    color = (0, 255, 255),
                    thickness = 2
                )

                def tracker_inside_in_region():
                    return point_in_rectangle(
                        point = (tracker_bottom_right_x, tracker_bottom_right_y),
                        rectangle = rectangle_in_region
                    )
                
                def tracker_inside_out_region():
                    return point_in_rectangle(
                        point = (tracker_top_left_x, tracker_top_left_y),
                        rectangle = rectangle_out_region
                    )

                if (count_incoming):
                    if (tracker_inside_in_region()):
                        in_count += 1
                        tracker_list.pop(i)
                        tracker_age_list.pop(i)
                if (count_outgoing):
                    if (tracker_inside_out_region()):
                        out_count += 1
                        tracker_list.pop(i)
                        tracker_age_list.pop(i)
            gt.stamp(name = "Draw bounding box & delete trackers", unique = False)
            
            colour_ui_box, colour_ui_text = (0, 255, 255), (0, 255, 255)
            opacity_ui_box, opacity_ui_text = 0.3, 0.3
            thickness = 2

            if (show_video_debug_info):
                # Rectangle of interest
                # frame_copy = put_rectangle_opaque(
                #     frame = frame_copy,
                #     point_1 = np.array(rectangle_of_interest[0]) + thickness,
                #     point_2 = np.array(rectangle_of_interest[1]) - thickness,
                #     colour = colour_ui_box,
                #     opacity = opacity_ui_box,
                #     thickness = thickness
                # )
                frame_copy = put_rectangle_with_title(
                    frame = frame_copy,
                    point_1 = np.array(rectangle_of_interest[0]) + thickness,
                    point_2 = np.array(rectangle_of_interest[1]) - thickness,
                    title = "Region of interest",
                    box_colour = colour_ui_box,
                    opacity = opacity_ui_box,
                    thickness = thickness,
                    title_colour = colour_ui_text,
                    font_size = 0.3,title_position = "ne"
                )

                if (count_incoming):
                    # In-region rectangle
                    frame_copy = put_rectangle_with_title(
                        frame = frame_copy, 
                        point_1 = rectangle_in_region[0], point_2 = rectangle_in_region[1],
                        title = "In-region", box_colour = colour_ui_box, opacity = opacity_ui_box,
                        thickness = -1, title_colour = colour_ui_text, 
                        font_size = 0.3, title_position = "ne"
                    )

                if (count_outgoing):
                    # Out-region rectangle
                    frame_copy = put_rectangle_with_title(
                        frame = frame_copy, 
                        point_1 = rectangle_out_region[0], point_2 = rectangle_out_region[1],
                        title = "Out-region", box_colour = colour_ui_box, opacity = opacity_ui_box,
                        thickness = -1, title_colour = colour_ui_text, 
                        font_size = 0.3, title_position = "ne"
                    )
            
            if (count_incoming):
                cv2.putText(
                    img = frame_copy,
                    text = "IN: {}".format(in_count),
                    org = (20, 40),
                    fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1, 
                    color = colour_ui_text, thickness = 2
                )
            if (count_outgoing):
                cv2.putText(
                    img = frame_copy,
                    text = "OUT: {}".format(out_count),
                    org = (frame_width - 100, 40),
                    fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1, color = colour_ui_text, thickness = 2
                )
            
            gt.stamp(name = "Other UI stuff", unique = False)

            if (generate_video_side_by_side):
                frame_concatenated = Image.new(
                    mode = 'RGB',
                    size = ((2 * frame_width) + ((2 * border_width) + width_inter_image), frame_height + (2 * border_height))
                )

                x_offset, y_offset = border_width, border_height
                frame_concatenated.paste(Image.fromarray(obj = (BGR_to_RGB(frame))), (x_offset, y_offset))
                x_offset += (frame_width + width_inter_image)
                frame_concatenated.paste(Image.fromarray(obj = (BGR_to_RGB(frame_copy))), (x_offset, y_offset))

                frame_list_side_by_side.append(np.asarray(frame_concatenated))
            frame_list_single.append(BGR_to_RGB(frame_copy))

            gt.stamp(name = "Prepare final video frame", unique = False)

    gt.stop()
    if (show_debug_info): print(gt.report())

    return (
        frame_list_single, frame_list_side_by_side
    ) if (generate_video_side_by_side) else frame_list_single

def create_video_from_frames(frame_list, video_name, fps = 30):
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError:
        %pip install imageio_ffmpeg

    import moviepy.video.io.ImageSequenceClip

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(sequence = frame_list, fps = fps)
    clip.write_videofile(filename = os.path.join(path_video, video_name))
    clip.close()
    
    return os.path.join(path_video, video_name)

def show_video(video_path):
    from IPython.display import HTML
    from base64 import b64encode

    mp4 = open(video_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    HTML(
        data = """
        <video width = 900 controls>
        <source src = "{}" type="video/mp4">
        </video>
        """.format(data_url)
    )

def main():
    generate_video_side_by_side = True

    result = generate_output_frames(
        count_incoming = True, count_outgoing = True,
        set_regions_automatically = False, show_debug_info = True, 
        show_video_debug_info = True, generate_video_side_by_side = generate_video_side_by_side
    )

    if (generate_video_side_by_side):
        frame_list_single, frame_list_side_by_side = result
        create_video_from_frames(frame_list = frame_list_single, video_name = "video_output_single.mp4", fps = 30)
        create_video_from_frames(frame_list = frame_list_side_by_side, video_name = "video_output_side_by_side.mp4", fps = 30)
    else:
        frame_list_single = result
        create_video_from_frames(frame_list = frame_list_single, video_name = "video_output_single.mp4", fps = 30)

    # show_video(video_path = os.path.join(path_video, "video_output.mp4"))

if (__name__ == "__main__"):
    main()