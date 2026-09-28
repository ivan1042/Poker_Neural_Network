def position_mapping(df, y = "position_norm"):
    POSITION_MAP = {
        0.000: "BTN",
        0.167: "SB",
        0.333: "BB",
        0.500: "UTG",
        0.667: "HJ",
        0.833: "CO",
    }

    temp = df[y].round(3).map(POSITION_MAP)
    df[y] = temp
    return df

def x_y_separator(df, y = "action"):
    label = df[y]
    df.drop(columns=[y], inplace=True)
    return df, label

def action_label(df):
    df.loc[df["if_fold"] == 1, "action"] = "fold"
    df.loc[(df["if_fold"] == 0) & (df["hero_bet_ratio"] == 0), "action"] = "check"
    df.loc[(df["facing_ratio"] < df["hero_bet_ratio"]) & (df["action"] != 0) & (
                df["action"] != 1), "action"] = "raise"
    df.fillna({"action": "call"}, inplace=True)

    return df

