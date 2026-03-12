import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import random
from PIL import Image
import json  # JSON読み込み用

ACTIVITIES = ['2p-succ.', '2p-fail.-off.', '2p-fail.-def.',
              '2p-layup-succ.', '2p-layup-fail.-off.', '2p-layup-fail.-def.',
              '3p-succ.', '3p-fail.-off.', '3p-fail.-def.']


def read_ids(path):
    with open(path) as file:
        values = file.readline()
    values = values.split(',')[:-1]
    values = list(map(int, values))
    return values


def nba_read_annotations(path, seqs):
    labels = {}
    group_to_id = {name: i for i, name in enumerate(ACTIVITIES)}

    for sid in seqs:
        annotations = {}
        with open(path + '/%d/annotations.txt' % sid) as f:
            for line in f.readlines():
                values = line.strip().split('\t')
                file_name = values[0]
                fid = int(file_name.split('.')[0])
                activity = group_to_id[values[1]]
                annotations[fid] = {
                    'file_name': file_name,
                    'group_activity': activity,
                }
            labels[sid] = annotations

    return labels


def nba_all_frames(labels):
    frames = []

    for sid, anns in labels.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))

    return frames

class NBADataset(data.Dataset):
    def __init__(self, frames, anns, image_path, args, is_training=True, bbox_path=None, tracking_path=None, net_path=None):
        super(NBADataset, self).__init__()
        self.frames = frames
        self.anns = anns
        self.backbone = args.backbone
        self.vit_arch = args.ViT_arch
        self.track_path = tracking_path
        self.tracks = self.load_people_tracks()
        self.image_path = image_path
        self.ball_annotation_path = bbox_path
        self.net_path = net_path
        self.image_size = (args.image_width, args.image_height)
        self.image_width = args.image_width
        self.image_height = args.image_height
        self.random_sampling = args.random_sampling
        self.num_frame = args.num_frame
        self.num_total_frame = args.num_total_frame
        self.num_boxes = 10
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
        self.max_flow_frame = 71  
        self.ball_annotations = self.load_ball_annotations()
        self.net_bboxes = self.load_net_bboxes()
        self.use_flow = args.use_flow
        self.use_flow_numpy = args.use_flow_numpy
        
    def load_people_tracks(self):
        tracks = {}
        for vid in self.anns.keys():
            for sid in self.anns[vid].keys():
                track_file = os.path.join(self.track_path, f'{vid}', f'{sid}', f'{vid}_{sid}.txt')
                if not os.path.exists(track_file):
                    print("Warning: Tracking file not found:", track_file)
                    continue

                tracks[(vid, sid)] = {}
                with open(track_file, 'r') as f:
                    lines = f.read().strip().splitlines()

                for line in lines:
                    values = line.strip().split(',')
                    if len(values) < 6:
                        continue
                    actual_fid = int(values[0])
                    x = float(values[2])
                    y = float(values[3])
                    w = float(values[4])
                    h = float(values[5])
                    
                    y1, x1, y2, x2 = y, x, y + h, x + w
                    bbox = [y1, x1, y2, x2]
                    
                    if actual_fid not in tracks[(vid, sid)]:
                        tracks[(vid, sid)][actual_fid] = []
                    tracks[(vid, sid)][actual_fid].append(bbox)
        return tracks

    def load_ball_annotations(self):
        ball_annotations = {}
        for vid in self.anns.keys():
            ball_annotations[vid] = {}
            for sid in self.anns[vid].keys():
                ball_annotations[vid][sid] = {}
                ball_file = os.path.join(self.ball_annotation_path, '%d' % vid, '%d.txt' % sid)
                if not os.path.exists(ball_file):
                    print("Warning: Ball coordinate file not found:", ball_file)
                    continue

                image_dir = os.path.join(self.image_path, '%d' % vid, '%d' % sid)
                if not os.path.exists(image_dir):
                    print("Warning: Image directory not found:", image_dir)
                    continue

                file_names = sorted(os.listdir(image_dir), key=lambda x: int(os.path.splitext(x)[0]))
                sids = [int(os.path.splitext(x)[0]) for x in file_names]

                with open(ball_file, 'r') as f:
                    lines = f.read().strip().splitlines()

                if len(lines) != len(sids):
                    print("Warning: Number of lines in ball file does not match number of images for vid {} sid {}.".format(vid, sid))

                coords = {}
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    y_str, x_str = parts[0], parts[1]
                    y = 0 if y_str == "-inf" else int(float(y_str))
                    x = 0 if x_str == "-inf" else int(float(x_str))
                    if i < len(sids):
                        current_sid = sids[i]
                    else:
                        current_sid = sids[0] + i
                    coords[current_sid] = (y, x)
                ball_annotations[vid][sid] = coords
        return ball_annotations
    
    def load_net_bboxes(self):
        net_bboxes = {}

        if self.net_path is None:
            return net_bboxes

        for vid in self.anns.keys():
            net_bboxes[vid] = {}
            for sid in self.anns[vid].keys():
                net_bboxes[vid][sid] = {}
                for fid in range(self.max_flow_frame):
                    fid_str = "{0:06d}".format(fid)
                    net_file = os.path.join(self.net_path, f"{vid}", f"{sid}", fid_str)

                    if not os.path.exists(net_file):
                        continue

                    try:
                        with open(net_file, "r") as f:
                            objs = json.load(f)
                    except Exception as e:
                        print(f"Warning: Failed to read net file {net_file}: {e}")
                        continue

                    candidates = [
                        o for o in objs
                        if o.get("name") == "Net" or o.get("class") == 1
                    ]
                    if len(candidates) == 0:
                        continue

                    best = max(candidates, key=lambda o: o.get("confidence", 0.0))
                    box = best.get("box", {})
                    x1 = float(box.get("x1", 0.0))
                    y1 = float(box.get("y1", 0.0))
                    x2 = float(box.get("x2", 0.0))
                    y2 = float(box.get("y2", 0.0))

                    net_bboxes[vid][sid][fid] = (x1, y1, x2, y2)

        return net_bboxes
    
    def __getitem__(self, idx):
        frames = self.select_frames(self.frames[idx])
        samples = self.load_samples(frames)
        return samples

    def __len__(self):
        return len(self.frames)

    def select_frames(self, frame):
        """
        The maximum frame index for which optical flow exists is self.max_flow_frame (71), so the selected frames will be within [0, self.max_flow_frame-1].
        """
        vid, sid = frame
        max_frame = self.max_flow_frame  # 71
        if self.is_training:
            if self.random_sampling == 'random_samp':
                sample_frames = random.sample(range(max_frame), self.num_frame)
                sample_frames.sort()
            elif self.random_sampling == 'full_frames':
                sample_frames = list(range(max_frame))
            elif self.random_sampling == 'fixed_interval':
                if self.num_frame == 6:
                    sample_frames = list(range(6, max_frame, 12))
                elif self.num_frame == 12:
                    sample_frames = list(range(4, max_frame, 6))
                elif self.num_frame == 18:
                    sample_frames = list(range(2, max_frame, 4))
                elif self.num_frame == max_frame:
                    sample_frames = list(range(max_frame))
                elif self.num_frame == 1:
                    sample_frames = [35 if 35 < max_frame else max_frame // 2]
                else:
                    segment_duration = max_frame // self.num_frame
                    sample_frames = (np.multiply(list(range(self.num_frame)), segment_duration) +
                                     segment_duration // 2)
                    sample_frames = [min(f, max_frame - 1) for f in sample_frames]
            else:
                segment_duration = max_frame // self.num_frame
                sample_frames = (np.multiply(list(range(self.num_frame)), segment_duration) +
                                 np.random.randint(segment_duration, size=self.num_frame))
                sample_frames = [min(f, max_frame - 1) for f in sample_frames]
        else:
            if self.num_frame == 3:
                sample_frames = list(range(12, max_frame, 24))
            elif self.num_frame == 6:
                sample_frames = list(range(6, max_frame, 12))
            elif self.num_frame == 12:
                sample_frames = list(range(4, max_frame, 6))
            elif self.num_frame == 18:
                sample_frames = list(range(2, max_frame, 4))
            elif self.num_frame == max_frame:
                sample_frames = list(range(max_frame))
            elif self.num_frame == 1:
                sample_frames = [35 if 35 < max_frame else max_frame // 2]
            else:
                segment_duration = max_frame // self.num_frame
                sample_frames = (np.multiply(list(range(self.num_frame)), segment_duration) +
                                 segment_duration // 2)
                sample_frames = [min(f, max_frame - 1) for f in sample_frames]
        return [(vid, sid, fid) for fid in sample_frames]

    def load_samples(self, frames):
        images, activities = [], []
        ball_coords = []
        net_boxes_list = []
        if self.use_flow or self.use_flow_numpy:
            optical_flow = []
        boxes, boxes_idx = [], []
        
        for i, (vid, sid, fid) in enumerate(frames):
            fid_str = '{0:06d}'.format(fid)
            img_path = os.path.join(self.image_path, f'{vid}', f'{sid}', f'{fid_str}.jpg')
            img = Image.open(img_path)
            original_w, original_h = img.size
            img = self.transform(img)
            images.append(img)
            activities.append(self.anns[vid][sid]['group_activity'])
            if self.use_flow:
                flow_dir = self.image_path.replace('videos', 'flow_min_max')
                flow_path = os.path.join(flow_dir, f'{vid}', f'{sid}', f'{fid_str}_flow.jpg')
                flow = Image.open(flow_path)
                flow = self.transform_flow(flow)
                flow = flow[1:3, :, :]
                optical_flow.append(flow)
            if self.use_flow_numpy:
                if tuple(self.image_size) in {(448, 252), (512, 288), (224, 224), (256, 256)}:
                    flow_dir = self.image_path.replace('videos', 'flow_numpy_sub_med')
                elif self.image_size == (896, 504) or self.image_size == (1024, 576): 
                    flow_dir = self.image_path.replace('videos', 'flow_numpy_sub_med_36x64')
                flow_path = os.path.join(flow_dir, f'{vid}', f'{sid}', f'{fid_str}_flow.npy')
                flow = np.load(flow_path)
                flow = torch.tensor(flow, dtype=torch.float)
                optical_flow.append(flow)

            if vid in self.ball_annotations and sid in self.ball_annotations[vid]:
                coords_dict = self.ball_annotations[vid][sid]
                if fid in coords_dict:
                    ball_coord = coords_dict[fid]
                else:
                    ball_coord = (0.0, 0.0)
                    print(f"Warning: Ball coordinate not found for vid {vid}, sid {sid}, fid {fid}.")
            else:
                ball_coord = (0.0, 0.0)
                print("Warning: Ball coordinate not found for vid {}.".format(vid))
            
            norm_x = ball_coord[0] / original_w
            norm_y = ball_coord[1] / original_h
            normalized_ball_coord = (norm_x, norm_y)
            ball_coords.append(torch.tensor(normalized_ball_coord, dtype=torch.float))
            
            if (
                hasattr(self, "net_bboxes") and
                vid in self.net_bboxes and
                sid in self.net_bboxes[vid] and
                fid in self.net_bboxes[vid][sid]
            ):
                x1, y1, x2, y2 = self.net_bboxes[vid][sid][fid]
                nx1 = x1 / original_w
                ny1 = y1 / original_h
                nx2 = x2 / original_w
                ny2 = y2 / original_h
                net_box = torch.tensor([nx1, ny1, nx2, ny2], dtype=torch.float)
            else:
                net_box = torch.zeros(4, dtype=torch.float)

            net_boxes_list.append(net_box)
            
            if (vid, sid) in self.tracks and fid in self.tracks[(vid, sid)]:
                track_list = self.tracks[(vid, sid)][fid]
            else:
                track_list = []

            if len(track_list) == 0:
                temp_boxes = np.zeros((self.num_boxes, 4), dtype=np.float32)
            else:
                temp_boxes = np.ones((len(track_list), 4), dtype=np.float32)
                for j, track in enumerate(track_list):
                    OW, OH = 1280.0, 720.0
                    y1, x1, y2, x2 = track
                    y1, x1, y2, x2 = y1 / original_h, x1 / original_w, y2 / original_h, x2 / original_w
                    w1, h1, w2, h2 = int(x1 * OW), int(y1 * OH), int(x2 * OW), int(y2 * OH)
                    w1, h1, w2, h2 = min(w1, OW-1), min(h1, OH-1), min(w2, OW-1), min(h2, OH-1)
                    temp_boxes[j] = np.array([w1, h1, w2, h2])
                if len(track_list) < self.num_boxes:
                    padding = np.tile(temp_boxes[0:1, :], (self.num_boxes - len(track_list), 1))
                    temp_boxes = np.vstack([temp_boxes, padding])
                elif len(track_list) > self.num_boxes:
                    temp_boxes = temp_boxes[:self.num_boxes, :]
            boxes.append(temp_boxes)
            boxes_idx.append(i * np.ones(self.num_boxes, dtype=np.int32))
            
        images = torch.stack(images)
        activities = torch.tensor(activities, dtype=torch.long)
        ball_coords = torch.stack(ball_coords)
        net_boxes = torch.stack(net_boxes_list)
        if self.use_flow or self.use_flow_numpy:
            optical_flow = torch.stack(optical_flow)
        bboxes = np.vstack(boxes).reshape([-1,self.num_boxes,4])
        bboxes = np.round(bboxes).astype(np.int32)
        bboxes = torch.tensor(bboxes, dtype=torch.float)
        bboxes_idx = np.hstack(boxes_idx).reshape([-1,self.num_boxes])
        bboxes_idx = torch.tensor(bboxes_idx, dtype=torch.int32)
        
        if self.use_flow or self.use_flow_numpy:
            return images, activities, ball_coords, optical_flow, bboxes, bboxes_idx, net_boxes
        else:
            return images, activities, ball_coords, bboxes, bboxes_idx, net_boxes