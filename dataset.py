import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import VivitImageProcessor

import cv2

class NFLDataset(Dataset):
    def __init__(self, labels, helmets, metadata, video_dir, img_size,
                 start, stop, interval):
        self.labels = labels
        self.helmets = helmets
        self.meta = metadata
        self.video_dir = video_dir
        self.img_size = img_size
        self.start = start
        self.stop = stop
        self.interval = interval
        self.img_preproc = VivitImageProcessor.from_pretrained('./models/.')

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        view = 'Sideline' if np.random.randint(2) == 0 else 'Endzone'
        gp = self.labels.iloc[idx, 1]
        step = self.labels.iloc[idx, 3]
        p1_id = self.labels.iloc[idx, 4]
        p2_id = self.labels.iloc[idx, 5]
        target = self.labels.iloc[idx, 6] 
        
        helms = join_helmets_step(gp, self.labels, self.helmets, self.meta)

        raw_frames = frame_extraction(
            self.video_dir,
            helms,
            gp,
            view,
            step,
            p1_id,
            p2_id,
            self.start,
            self.stop,
            self.interval,
            self.img_size
        )
        
        processed_frames = self.img_preproc(list(raw_frames), return_tensors="pt")
        processed_frames = torch.squeeze(processed_frames['pixel_values'])
        return raw_frames, processed_frames, target
    

def join_helmets_step(game_play, labels, helmets, meta, fps=59.94):
    """
    Selects helmet data for specified game_play and adds step column (timestep)
    Returns new game_play-specific helmet + step dataframe 
    """
    gp_helms = helmets[helmets['game_play'] == game_play].copy()

    start_time = meta[meta['game_play'] == game_play]['start_time'].values[0]

    gp_helms['datetime'] = pd.to_datetime(
        pd.to_timedelta(gp_helms['frame'] * (1 / fps), unit='s') + start_time,
        utc=True
    )
    gp_helms['datetime_ngs'] = pd.to_datetime(
        pd.DatetimeIndex(gp_helms['datetime'] + pd.to_timedelta(50, 'ms'))
        .floor('100ms')
        .values,
        utc=True
    )

    gp_labs = labels[labels['game_play'] == game_play][
        ['datetime', 'step']
    ].copy()
    gp_labs.drop_duplicates(inplace=True)
    gp_labs['datetime'] = pd.to_datetime(gp_labs['datetime'], utc=True, format='mixed')

    gp_helms_w_step = gp_helms.merge(
        gp_labs,
        left_on=['datetime_ngs'],
        right_on=['datetime'],
        how='left'
    )
    return gp_helms_w_step


def add_masks(frame, p1_pos, p2_pos):
    """
    Adds white/black circle over Player 1/2's helmet position, respectively
    Returns masked frame
    """
    p1_pos = p1_pos.iloc[0]
    x1 = p1_pos['left'] + int(p1_pos['width'] / 2)
    y1 = p1_pos['top'] + int(p1_pos['height'] / 2)
    r1 = int(((p1_pos['height'] + p1_pos['width']) / 2) / 2)
    frame = cv2.circle(frame, (x1, y1), r1, (255, 255, 255), -1)
    if not p2_pos.empty:
        p2_pos = p2_pos.iloc[0]
        x2 = p2_pos['left'] + int(p2_pos['width'] / 2)
        y2 = p2_pos['top'] + int(p2_pos['height'] / 2)
        r2 = int(((p2_pos['height'] + p2_pos['width']) / 2) / 2)
        frame = cv2.circle(frame, (x2, y2), r2, (0, 0, 0), -1)
    return frame


def crop_frame(frame, p1_pos, p2_pos):
    """
    Crops frame around Players 1/2 with padding between player & frame edge
    Returns cropped frame
    """
    height, width = frame.shape[0], frame.shape[1]
    left, right, top, bottom = 0, 0, 0, 0
    if not p2_pos.empty:
        p1_pos, p2_pos = p1_pos.iloc[0], p2_pos.iloc[0]
        left = max(min(p1_pos['left'], p2_pos['left']) - 75, 0)
        right = min(max(p1_pos['left'] + p1_pos['width'], 
                        p2_pos['left'] + p2_pos['width']) + 75, width)
        top = max(min(p1_pos['top'], p2_pos['top']) - 75, 0)
        bottom = min(max(p1_pos['top'] + p1_pos['height'],
                        p2_pos['top'] + p2_pos['height']) + 75, height)
    else:
        p1_pos = p1_pos.iloc[0]
        left = max(p1_pos['left'] - 100, 0)
        right = min(p1_pos['left'] + p1_pos['width'] + 100, width)
        top = max(p1_pos['top'] - 100, 0)
        bottom = min(p1_pos['top'] + p1_pos['height'] + 100, height)
    frame = frame[top:bottom, left:right]
    return frame          


def frame_extraction(video_dir, helmets, game_play, view, step, p1_id, p2_id, 
                     start, stop, interval, img_size):
    """
    Extracts four frames before/after a given timestep (step) in a video
    Masks Player 1/2 helmets on each frame and crops frame around them
    Returns a 9 frame long sequence of masked/cropped frames
    """
    helms = helmets[helmets['view'] == view].copy()

    frame_ids = np.array(range(start, stop, interval))
    frames_list = np.zeros((len(frame_ids), img_size, img_size, 3))

    middle_frame = 0
    if step in helmets['step'].unique():
        middle_frame = min(helmets[helmets['step'] == step]['frame'])
    elif (step - 1) in helmets['step'].unique():
        middle_frame = max(helmets[helmets['step'] == step]['frame']) + 1
    elif (step + 1) in helmets['step'].unique():
        middle_frame = min(helmets[helmets['step'] == step]['frame']) - 5
    else:
        return frames_list

    # Path to video & VideoCapture object to read video
    video_path = video_dir + game_play + '_' + view + '.mp4'
    video_reader = cv2.VideoCapture(video_path)
  
    for i, x in enumerate(frame_ids):
        # Frame to extract from video
        current_frame = middle_frame + x
        # Filter helmet bounding box data based using specific frame and view 
        helms_filtered = helms[helms['frame'] == current_frame]
        # Extract current_frame from game_play video 
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, frame = video_reader.read() 
        if not success:
            continue

        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Player 1/2's helmet positions (if in frame)
        p1_pos = helms_filtered[helms_filtered['nfl_player_id'] == p1_id]
        p2_pos = helms_filtered[helms_filtered['nfl_player_id'] == int(p2_id)] \
            if p2_id != 'G' else pd.DataFrame()
        # Add helmet masks & crop frame
        if not p1_pos.empty:
            frame = add_masks(frame, p1_pos, p2_pos)
            frame = crop_frame(frame, p1_pos, p2_pos)
        # Resize the frame, normalize pixel values & add to frames_list
        frame = cv2.resize(frame, (img_size, img_size))
        frames_list[i] = frame
    
    video_reader.release()
    return np.array(frames_list)