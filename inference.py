import os
import yaml
import numpy as np
import pandas as pd

import torch
from transformers import VivitImageProcessor, VivitForVideoClassification
from torch.utils.data import Dataset, DataLoader

from train import load_config
from dataset import frame_extraction, join_helmets_step


class InferenceDataset(Dataset):
    def __init__(self, ss, tracking, helmets, metadata, img_size,
                 start, stop, interval):
        self.ss = ss
        self.tracking = tracking
        self.helmets = helmets
        self.meta = metadata
        self.video_dir = 'data/test/'
        self.img_size = img_size
        self.start = start
        self.stop = stop
        self.interval = interval
        self.img_preproc = VivitImageProcessor.from_pretrained('./models/.')

    def __len__(self):
        return len(self.ss)
    
    def __getitem__(self, idx):
        cid = self.ss.iloc[idx].contact_id
        s = cid.split('_')
        gp = s[0] + '_' + s[1]
        step = int(s[2])
        p1 = int(s[3])
        p2 = s[4]
        view = 'Sideline'
        helms = join_helmets_step(gp, self.tracking, self.helmets, self.meta)

        raw_frames = frame_extraction(
            self.video_dir,
            helms,
            gp,
            view,
            step,
            p1,
            p2,
            self.start,
            self.stop,
            self.interval,
            self.img_size
        )
        
        processed_frames = self.img_preproc(list(raw_frames), return_tensors="pt")
        processed_frames = torch.squeeze(processed_frames['pixel_values'])
        return cid, processed_frames


def compute_distance(df, te_tracking, merge_col="step"):
    """
    Merges tracking data on player1 and 2 and computes the distance.
    """
    df_combo = df.merge(
        te_tracking.astype({"nfl_player_id": "str"})[
            ["game_play", merge_col, "nfl_player_id", "x_position", "y_position"]
        ],
        left_on=['game_play', 'nfl_player_id_1', merge_col],
        right_on=['game_play', 'nfl_player_id', merge_col],
        how='left'
    ).rename(columns={"x_position": "x_position_1", "y_position": "y_position_1"}) \
        .drop("nfl_player_id", axis=1)\
        .merge(
            te_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id", "x_position", "y_position"]
            ],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        ).drop("nfl_player_id", axis=1)\
            .rename(columns={"x_position": "x_position_2", "y_position": "y_position_2"})\
                .copy()


    df_combo["distance"] = np.sqrt(
        np.square(df_combo["x_position_1"] - df_combo["x_position_2"])
        + np.square(df_combo["y_position_1"] - df_combo["y_position_2"])
    )
    return df_combo


def predict(model, dataloader, device):
    model.to(device)
    model.eval()
    contact_ids, preds = [], []
    for i, (cid, data) in enumerate(dataloader):
        if i % 10 == 0: print(i)
        output = model(pixel_values=data.to(device))
        logits = output.logits
        preds += logits.argmax(-1)
        contact_ids += list(cid)
    return contact_ids, preds


if __name__ == '__main__':
    print('Loading configs & data')
    cnf = load_config('', 'config.yaml')
    helmets = pd.read_csv(f'./data/test_baseline_helmets.csv')
    tracking = pd.read_csv('./data/test_player_tracking.csv', parse_dates=['datetime'])
    metadata = pd.read_csv('./data/test_video_metadata.csv',
                                    parse_dates=["start_time", "end_time", "snap_time"])
    ss = pd.read_csv('./data/sample_submission.csv')
    ## Compute distance between players ##
    p1, p2, step, gp = [],[],[],[]
    for s in ss['contact_id'].values:
        a = s.split('_')
        gp.append(a[0] + '_' + a[1])
        step.append(a[2])
        p1.append(a[3])
        p2.append(a[4])

    d = {"nfl_player_id_1":p1, "nfl_player_id_2":p2, 'step':step, 'game_play':gp}
    df = pd.DataFrame(data=d)
    df = df.astype({"nfl_player_id_1": "str", "nfl_player_id_2":'str', 'step':'int', 'game_play':'str'})
    
    df_combo = compute_distance(df, tracking)
    df_dist = df_combo.merge(tracking[["game_play", "step"]].drop_duplicates())
    df_dist["distance"] = df_dist["distance"].fillna(99) 
    df_dist["contact_id"] = (df_dist["game_play"] + "_" + 
                             df_dist["step"].astype("str") + "_" +
                             df_dist["nfl_player_id_1"].astype("str") + "_" + 
                             df_dist["nfl_player_id_2"].astype("str"))
    #####################################
    ## Filter players that are too far to be in contact with one another **
    possible_contact = df_dist[(df_dist['distance'] <= 3) | (df_dist['distance'] == 99)].copy()
    no_contact = list(df_dist[(df_dist['distance'] > 3) & (df_dist['distance'] != 99)].contact_id)
    
    contact_ids = no_contact
    predictions = [0]*len(contact_ids)
    #####################################

    print('Creating dataloader')
    inf_dataset = InferenceDataset(possible_contact, tracking, helmets, metadata, 
                                   cnf['img_size'], cnf['start'], cnf['stop'], cnf['interval'])
    inf_dataloader = DataLoader(inf_dataset, batch_size=cnf['batch_size'])
    
    print('Loading model')
    model = VivitForVideoClassification.from_pretrained(
        './models/.',
        num_frames=cnf['stop'],
        num_labels=cnf['n_labels'],
        ignore_mismatched_sizes=True
    )

    print('Making predictions')
    c, p = predict(model, inf_dataloader, cnf['device'])
    contact_ids += c
    predictions += p

    print('Saving predictions')
    d = {'contact_id': contact_ids, 'contact': predictions}
    df = pd.DataFrame(data=d)
    df.to_csv('predictions.csv')