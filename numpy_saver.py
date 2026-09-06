from sample3 import FeatureExtractor
import numpy as np
import pandas as pd
from pathlib import Path
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
name = "sample_poker_hand_history.txt"
def hand_updater(new_hand_folder: str):
    #r"C:\Users\user\Downloads\0000019c-e65b-d101-0000-0000edc2a1d4"
    folder = Path(new_hand_folder)
    temp = []
    for file_path in folder.glob("*.txt"):
        with open(file_path, "r") as f:
            content = f.read()
        test = content.split("\n\n\n")
        for k in test:
            extractor = FeatureExtractor(k)
            data = extractor.extract_data()
            for s in data:
                temp.append(s)
    df = pd.DataFrame(temp, columns=["My_hand", "stage", "community_card", "position", "(suit_po, stra_po, gut)", "facing_bet_ra", "if_fold", "my action", "result"])


    X_df = pd.DataFrame(index=df.index)

    # Hero hand (always 2)
    hero_ranks_df = pd.DataFrame(df['My_hand'].tolist(), index=df.index,
                                 columns=['hero_r1', 'hero_r2'])
    X_df = pd.concat([X_df, hero_ranks_df], axis=1)

    # Stage (always 4)
    stage_df = pd.DataFrame(df['stage'].tolist(), index=df.index,
                            columns=['stage_pre', 'stage_flop', 'stage_turn', 'stage_river'])
    X_df = pd.concat([X_df, stage_df], axis=1)

    # Board (already padded to 5 in your extractor)
    board_df = pd.DataFrame(df['community_card'].tolist(), index=df.index,
                            columns=[f'board_r{i+1}' for i in range(5)])
    X_df = pd.concat([X_df, board_df], axis=1)

    # Potentials — assuming it returns exactly 3 values for now
    # If the length varies → you must fix it in check_potential() or pad here
    potential_names = ['suit_po', 'stra_po', 'gut_po']   # update if more features
    potentials_df = pd.DataFrame(df["(suit_po, stra_po, gut)"].tolist(), index=df.index,
                                 columns=potential_names)
    X_df = pd.concat([X_df, potentials_df], axis=1)

    # Scalars
    X_df['position_norm']   = df['position']
    X_df['facing_ratio']    = df['facing_bet_ra']
    X_df['if_fold'] = df['if_fold']
    X_df['hero_bet_ratio']  = df['my action']   # consider zeroing when fold/call
    X_df['result'] = df['result']


    existing = pd.read_csv("parsed_save.csv", index_col=0)
    latest = pd.concat([existing, X_df], axis=0, ignore_index=True)
    latest.to_csv("parsed_save2.csv")

hand_updater(r"C:\Users\user\Downloads\0000019c-e65b-d101-0000-0000edc2a1d4")
