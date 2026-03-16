from .volleyball_bbox_flow_detector import *
from .nba_bbox_flow_detector import *
from .jrdb import *

import pickle

TRAIN_SEQS_VOLLEY = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
# TRAIN_SEQS_VOLLEY = [10]
VAL_SEQS_VOLLEY = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
# VAL_SEQS_VOLLEY = []
TEST_SEQS_VOLLEY = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
# TEST_SEQS_VOLLEY = [9]

def read_dataset(args):
    # 初期化
    train_set = None
    test_set = None
    data_path = None
    image_path = None
    test_id_path = None
    test_ids = None
    test_frames = None
    tracking_path = None

    if args.dataset == 'volleyball':
        data_path = args.data_path + args.dataset
        image_path = data_path + "/videos"
        if args.detector:
            bbox_path = data_path + "/wasb"
            tracking_path = data_path + "/volleyball_tracks_deep_eiou"
        else:
            bbox_path = data_path + "/volleyball-weak/volleyball_ball_annotation"
        track_path = data_path + "/tracks_normalized.pkl"
        all_tracks = pickle.load(open(track_path, 'rb'))

        train_data = volleyball_read_annotations(image_path,
                            TRAIN_SEQS_VOLLEY + VAL_SEQS_VOLLEY, args.num_activities)
        train_frames = volleyball_all_frames(train_data)
        # train_frames = [(10, 13980)]

        test_data = volleyball_read_annotations(image_path,
                            TEST_SEQS_VOLLEY, args.num_activities)
        test_frames = volleyball_all_frames(test_data)
        # test_frames = [(9, 25950)]
        
        train_set = VolleyballDataset(train_frames, all_tracks, train_data, image_path, args, is_training=True, ball_annotation_path=bbox_path, tracking_path=tracking_path)
        train_set_for_val = VolleyballDataset(train_frames, all_tracks, train_data, image_path, args, is_training=False, ball_annotation_path=bbox_path, tracking_path=tracking_path)
        test_set = VolleyballDataset(test_frames, all_tracks, test_data, image_path, args, is_training=False, ball_annotation_path=bbox_path, tracking_path=tracking_path)

    elif args.dataset == 'nba':
        data_path = args.data_path + 'NBA_dataset'
        image_path = data_path + "/videos"
        # bboxのルートパスを指定（例：NBA_dataset/nba/objects）
        #bbox_path = data_path + "/nba/objects"
        bbox_path = data_path + "/wasb"
        track_path = data_path + "/tracks_deep_eiou_prune_72"

        train_id_path = data_path + "/train_video_ids"
        # train_id_path = data_path + "/train_video_ids_5%_1"
        test_id_path = data_path + "/test_video_ids"

        train_ids = read_ids(train_id_path)
        test_ids = read_ids(test_id_path)

        train_data = nba_read_annotations(image_path, train_ids)
        train_frames = nba_all_frames(train_data)

        test_data = nba_read_annotations(image_path, test_ids)
        test_frames = nba_all_frames(test_data)
        # test_frames = [(21801165, 445)]

        # NBADataset生成時にbbox_pathを渡す（これにより各フレームでバスケットボールの座標も読み込む）
        train_set = NBADataset(train_frames, train_data, image_path, args, is_training=True, bbox_path=bbox_path, tracking_path=track_path)
        train_set_for_val = NBADataset(train_frames, train_data, image_path, args, is_training=False, bbox_path=bbox_path, tracking_path=track_path)
        test_set = NBADataset(test_frames, test_data, image_path, args, is_training=False, bbox_path=bbox_path, tracking_path=track_path)
    
    elif args.dataset == 'jrdb':
        annotation_path = args.data_path + 'jrdb_par/annotations'
        image_path = args.data_path + 'jrdb_par/videos'
        test_seqs = [2, 7, 11, 16, 17, 25, 26]
        trains_seqs = [s for s in range(0, 26) if s not in test_seqs]
        num_actions = 27
        num_activities = 7
        num_social_activities = 32
        num_boxes = 60
        image_size = args.image_height, args.image_width
        out_size = image_size
        num_frame = args.num_frame
        train_anns = jrdb_read_dataset_new(annotation_path, trains_seqs, num_actions, num_activities, num_social_activities)
        train_frames = jrdb_all_frames(train_anns)

        test_anns = jrdb_read_dataset_new(annotation_path, test_seqs, num_actions, num_activities, num_social_activities)
        test_frames = jrdb_all_frames(test_anns)

        training_set = JRDB_Dataset(num_actions, num_activities,
                                    num_social_activities, train_anns, train_frames,
                                    image_path, image_size, out_size, args, num_boxes=num_boxes,
                                    num_frame=num_frame, is_training=True,
                                    is_finetune=False)
        
        training_set_for_val = JRDB_Dataset(num_actions, num_activities,
                                    num_social_activities, train_anns, train_frames,
                                    image_path, image_size, out_size, args, num_boxes=num_boxes,
                                    num_frame=num_frame, is_training=False,
                                    is_finetune=False)

        validation_set = JRDB_Dataset(num_actions, num_activities,
                                    num_social_activities, test_anns, test_frames,
                                    image_path, image_size, out_size, args, num_boxes=num_boxes,
                                    num_frame=num_frame, is_training=False,
                                    is_finetune=False)

        print('Reading dataset finished...')
        print('%d train samples' % len(train_frames))
        print('%d test samples' % len(test_frames))

        return training_set, training_set_for_val, validation_set
    
    else:
        assert False

    print("%d train samples and %d test samples" % (len(train_frames), len(test_frames)))

    # return train_set, train_set_for_val, test_set, data_path, image_path, test_id_path, test_ids, test_frames
    return train_set, train_set_for_val, test_set, data_path, image_path, test_id_path, train_frames, test_frames