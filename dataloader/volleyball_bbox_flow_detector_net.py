import os
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']

def volleyball_read_annotations(path, seqs, num_activities):
    labels = {}
    if num_activities == 8:
        group_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    # merge pass/set label    
    elif num_activities == 6:
        group_to_id = {'r_set': 0, 'r_spike': 1, 'r-pass': 0, 'r_winpoint': 2,
                       'l_set': 3, 'l-spike': 4, 'l-pass': 3, 'l_winpoint': 5}
    
    for sid in seqs:
        annotations = {}
        with open(os.path.join(path, '%d/annotations.txt' % sid)) as f:
            for line in f.readlines():
                values = line.strip().split(' ')
                file_name = values[0]
                fid = int(file_name.split('.')[0])
                activity = group_to_id[values[1]]
                annotations[fid] = {
                    'file_name': file_name,
                    'group_activity': activity,
                }
            labels[sid] = annotations

    return labels

def volleyball_all_frames(labels):
    frames = []
    for sid, anns in labels.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))
    return frames


class VolleyballDataset(data.Dataset):
    def __init__(self, frames, tracks, anns, image_path, args, is_training=True, ball_annotation_path=None, tracking_path=None, net_annotation_path=None):
        super(VolleyballDataset, self).__init__()
        self.frames = frames
        self.tracks = tracks
        self.anns = anns
        self.backbone = args.backbone
        self.vit_arch = args.ViT_arch
        self.detector_mode = args.detector
        if self.detector_mode:
            self.track_path = tracking_path
            self.tracks = self.load_people_tracks()
        self.image_path = image_path
        self.ball_annotation_path = ball_annotation_path
        self.net_annotation_path = net_annotation_path
        self.image_size = (args.image_width, args.image_height)
        self.random_sampling = args.random_sampling
        self.num_frame = args.num_frame
        self.num_total_frame = args.num_total_frame
        self.num_boxes = 12
        self.is_training = is_training
        if args.backbone in ['clip', 'siglip', 'siglip2']:
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width)),
            transforms.ToTensor(),
            normalize,
        ])
        self.transform_flow = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.ball_annotations = self.load_ball_annotations()
        self.net_annotations = self.load_net_annotations()
        self.use_flow = args.use_flow
        self.use_flow_numpy = args.use_flow_numpy

    def load_people_tracks(self):
        tracks = {}
        for sid in self.anns.keys():
            for src_fid in self.anns[sid].keys():
                track_file = os.path.join(self.track_path, f'{sid}', f'{src_fid}', f'{sid}_{src_fid}.txt')
                if not os.path.exists(track_file):
                    print("Warning: Tracking file not found:", track_file)
                    continue

                tracks[(sid, src_fid)] = {}
                with open(track_file, 'r') as f:
                    lines = f.read().strip().splitlines()

                offset = int(src_fid) - 20

                for line in lines:
                    values = line.strip().split(',')

                    if len(values) < 6:
                        continue
                    relative_frame = int(values[0])
                    actual_fid = relative_frame + offset

                    x = float(values[2])
                    y = float(values[3])
                    w = float(values[4])
                    h = float(values[5])

                    if int(sid) in [2, 37, 38, 39, 40, 41, 44, 45]:
                        norm_y1 = y / 1080.0
                        norm_x1 = x / 1920.0
                        norm_y2 = (y + h) / 1080.0
                        norm_x2 = (x + w) / 1920.0
                    else:
                        norm_y1 = y / 720.0
                        norm_x1 = x / 1280.0
                        norm_y2 = (y + h) / 720.0
                        norm_x2 = (x + w) / 1280.0
                    bbox = [norm_y1, norm_x1, norm_y2, norm_x2]
                    
                    if actual_fid not in tracks[(sid, src_fid)]:
                        tracks[(sid, src_fid)][actual_fid] = []
                    tracks[(sid, src_fid)][actual_fid].append(bbox)
        return tracks


    def load_ball_annotations(self):
        ball_annotations = {}
        for sid in self.anns.keys():
            ball_annotations[sid] = {}
            for src_fid in self.anns[sid].keys():
                ball_file = os.path.join(self.ball_annotation_path, '%d' % sid, '%d.txt' % src_fid)
                if not os.path.exists(ball_file):
                    print("Warning: Ball coordinate file not found:", ball_file)
                    continue

                image_dir = os.path.join(self.image_path, '%d' % sid, '%d' % src_fid)
                if not os.path.exists(image_dir):
                    print("Warning: Image directory not found:", image_dir)
                    continue

                file_names = sorted(os.listdir(image_dir), key=lambda x: int(os.path.splitext(x)[0]))
                fids = [int(os.path.splitext(x)[0]) for x in file_names]

                with open(ball_file, 'r') as f:
                    lines = f.read().strip().splitlines()

                if len(lines) != len(fids):
                    print("Warning: Number of lines in ball file does not match number of images for sid {} src_fid {}.".format(sid, src_fid))

                coords = {}
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    y_str, x_str = parts[0], parts[1]
                    y = 0 if y_str == "-inf" else int(float(y_str))
                    x = 0 if x_str == "-inf" else int(float(x_str))
                    if i < len(fids):
                        current_fid = fids[i]
                    else:
                        current_fid = fids[0] + i
                    coords[current_fid] = (y, x)
                ball_annotations[sid][src_fid] = coords
        return ball_annotations
    
    def load_net_annotations(self):
        net_annotations = {}
        if self.net_annotation_path is None:
            return net_annotations

        for sid in self.anns.keys():
            for src_fid in self.anns[sid].keys():
                net_dir = os.path.join(self.net_annotation_path, f"{sid}", f"{src_fid}")
                if not os.path.exists(net_dir):
                    continue

                net_annotations[(sid, src_fid)] = {}

                image_dir = os.path.join(self.image_path, f"{sid}", f"{src_fid}")
                if not os.path.exists(image_dir):
                    continue

                file_names = sorted(os.listdir(image_dir), key=lambda x: int(os.path.splitext(x)[0]))
                fids = [int(os.path.splitext(x)[0]) for x in file_names]

                for fid in fids:
                    csv_path = os.path.join(net_dir, f"{fid}.csv")
                    if not os.path.exists(csv_path):
                        net_annotations[(sid, src_fid)][fid] = (0.0, 0.0, 0.0, 0.0)
                        continue

                    with open(csv_path, "r") as f:
                        lines = f.read().strip().splitlines()

                    if len(lines) < 2:
                        net_annotations[(sid, src_fid)][fid] = (0.0, 0.0, 0.0, 0.0)
                        continue

                    parts = lines[1].split(",")
                    if len(parts) < 4:
                        net_annotations[(sid, src_fid)][fid] = (0.0, 0.0, 0.0, 0.0)
                        continue

                    xmin = float(parts[0])
                    ymin = float(parts[1])
                    xmax = float(parts[2])
                    ymax = float(parts[3])

                    xmin = max(0.0, min(1.0, xmin))
                    ymin = max(0.0, min(1.0, ymin))
                    xmax = max(0.0, min(1.0, xmax))
                    ymax = max(0.0, min(1.0, ymax))

                    net_annotations[(sid, src_fid)][fid] = (xmin, ymin, xmax, ymax)

        return net_annotations

    def __getitem__(self, idx):
        frames = self.select_frames(self.frames[idx])
        samples = self.load_samples(frames)
        return samples

    def __len__(self):
        return len(self.frames)

    def select_frames(self, frame):
        sid, src_fid = frame

        if self.is_training:
            if self.random_sampling == 'random_samp':
                sample_frames = random.sample(range(src_fid - 5, src_fid + 5), self.num_frame)
                sample_frames.sort()
            else:
                segment_duration = self.num_total_frame // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) \
                                + np.random.randint(segment_duration, size=self.num_frame) \
                                + src_fid - segment_duration * (self.num_frame // 2)
        else:
            segment_duration = self.num_total_frame // self.num_frame
            sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) \
                            + src_fid - segment_duration * (self.num_frame // 2)

        return [(sid, src_fid, fid) for fid in sample_frames]

    def load_samples(self, frames):
        images, activities = [], []
        ball_coords = []
        if self.use_flow or self.use_flow_numpy:
            optical_flow = []
        boxes, boxes_idx = [], []
        net_boxes = []

        for i, (sid, src_fid, fid) in enumerate(frames):
            img_path = os.path.join(self.image_path, '%d' % sid, '%d' % src_fid, '%d.jpg' % fid)
            img = Image.open(img_path)
            img = self.transform(img)
            images.append(img)
            activities.append(self.anns[sid][src_fid]['group_activity'])
            if self.use_flow:
                flow_dir = self.image_path.replace('videos', 'flow_min_max')
                flow_path = os.path.join(flow_dir, '%d' % sid, '%d' % src_fid, '%d_flow.jpg' % fid)
                flow = Image.open(flow_path)
                flow = self.transform_flow(flow)
                flow = flow[1:3, :, :]
                optical_flow.append(flow)
            if self.use_flow_numpy:
                if tuple(self.image_size) in {(448, 252), (512, 288), (224, 224), (256, 256)}:
                    # flow_dir = self.image_path.replace('videos', 'flow_numpy')
                    flow_dir = self.image_path.replace('videos', 'flow_numpy_sub_med')
                elif self.image_size == (896, 504) or self.image_size == (1024, 576): 
                    flow_dir = self.image_path.replace('videos', 'flow_numpy_sub_med_36x64')
                flow_path = os.path.join(flow_dir, '%d' % sid, '%d' % src_fid, '%d_flow.npy' % fid)
                flow = np.load(flow_path)
                flow = torch.tensor(flow, dtype=torch.float)
                optical_flow.append(flow)
            if sid in self.ball_annotations and src_fid in self.ball_annotations[sid]:
                coords_dict = self.ball_annotations[sid][src_fid]
                if fid in coords_dict:
                    ball_coord = coords_dict[fid]
                else:
                    ball_coord = (0.0, 0.0)
            else:
                ball_coord = (0.0, 0.0)
            
            if sid in [2, 37, 38, 39, 40, 41, 44, 45]:
                norm_y = ball_coord[0] / 1920.0
                norm_x = ball_coord[1] / 1080.0
            else:
                norm_y = ball_coord[0] / 1280.0
                norm_x = ball_coord[1] / 720.0
            normalized_ball_coord = (norm_y, norm_x)
            ball_coords.append(torch.tensor(normalized_ball_coord, dtype=torch.float))
            
            if (sid, src_fid) in self.net_annotations and fid in self.net_annotations[(sid, src_fid)]:
                xmin, ymin, xmax, ymax = self.net_annotations[(sid, src_fid)][fid]
            else:
                xmin, ymin, xmax, ymax = (0.0, 0.0, 0.0, 0.0)

            net_boxes.append(torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float))

            if self.detector_mode:
                if (sid, src_fid) in self.tracks and fid in self.tracks[(sid, src_fid)]:
                    track_list = self.tracks[(sid, src_fid)][fid]
                else:
                    track_list = []

                if len(track_list) == 0:
                    temp_boxes = np.zeros((self.num_boxes, 4), dtype=np.float32)
                else:
                    temp_boxes = np.ones((len(track_list), 4), dtype=np.float32)
                    for j, track in enumerate(track_list):
                        OW, OH = 1280.0, 720.0
                        y1, x1, y2, x2 = track
                        w1, h1, w2, h2 = int(x1 * OW), int(y1 * OH), int(x2 * OW), int(y2 * OH)
                        w1, h1, w2, h2 = min(w1, int(OW)-1), min(h1, int(OH)-1), min(w2, int(OW)-1), min(h2, int(OH)-1)
                        temp_boxes[j] = np.array([w1, h1, w2, h2])
                    if len(track_list) < self.num_boxes:
                        padding = np.tile(temp_boxes[0:1, :], (self.num_boxes - len(track_list), 1))
                        temp_boxes = np.vstack([temp_boxes, padding])
                    elif len(track_list) > self.num_boxes:
                        temp_boxes = temp_boxes[:self.num_boxes, :]
                boxes.append(temp_boxes)
                boxes_idx.append(i * np.ones(self.num_boxes, dtype=np.int32))
            else:
                temp_boxes = np.ones_like(self.tracks[(sid, src_fid)][fid])
                for i, track in enumerate(self.tracks[(sid, src_fid)][fid]):
                    OW, OH = 1280.0, 720.0
                    y1, x1, y2, x2 = track
                    w1, h1, w2, h2 = int(x1 * OW), int(y1 * OH), int(x2 * OW), int(y2 * OH)
                    w1, h1, w2, h2 = min(w1, OW-1), min(h1, OH-1), min(w2, OW-1), min(h2, OH-1)
                    temp_boxes[i] = np.array([w1, h1, w2, h2])
                boxes.append(temp_boxes)
                if len(boxes[-1]) != self.num_boxes:
                    boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes-len(boxes[-1])]])
                boxes_idx.append(i * np.ones(self.num_boxes, dtype=np.int32))

        images = torch.stack(images)
        activities = torch.tensor(activities, dtype=torch.long)
        ball_coords = torch.stack(ball_coords)  # shape: (num_frame, 2)
        if self.use_flow or self.use_flow_numpy:
            optical_flow = torch.stack(optical_flow)
        bboxes = np.vstack(boxes).reshape([-1,self.num_boxes,4])
        bboxes = np.round(bboxes).astype(np.int32)
        bboxes = torch.tensor(bboxes, dtype=torch.float)
        bboxes_idx = np.hstack(boxes_idx).reshape([-1,self.num_boxes])
        bboxes_idx = torch.tensor(bboxes_idx, dtype=torch.int32)
        net_boxes = torch.stack(net_boxes)
        
        if self.use_flow or self.use_flow_numpy:
            return images, activities, ball_coords, optical_flow, bboxes, bboxes_idx, net_boxes

        else:
            return images, activities, ball_coords, bboxes, bboxes_idx, net_boxes
