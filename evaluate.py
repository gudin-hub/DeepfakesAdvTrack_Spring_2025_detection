import os
import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

def eval_func(gt_name="gts", path=""):
    ret = {}

    teams = os.listdir(path)
    teams.remove(gt_name + ".xlsx")
    if "LeaderBoard.xlsx" in teams:
        teams.remove("LeaderBoard.xlsx")

    gts = pd.read_excel(
            os.path.join(path, gt_name + ".xlsx"),
        )["labels"]
    gts = np.array(gts)
    
    for team in teams:
        data = pd.read_excel(
            os.path.join(path, team),
            sheet_name="predictions",
        )
        preds = data["predictions"]
        preds = np.array(preds)
        auc = roc_auc_score(gts, preds)

        data = pd.read_excel(
            os.path.join(path, team),
            sheet_name="time",
        )
        mean_time = data["Time"][0] / data["Data Volume"][0]
        ret[team.split(".")[0]] = [auc, mean_time]

    return ret
    
if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--submit_path",
        default= './result',
        type=str
    )
    opts = arg.parse_args()
    results = eval_func(path=opts.submit_path)

    writer = pd.ExcelWriter(os.path.join(opts.submit_path, "LeaderBoard.xlsx"))
    prediction_frame = pd.DataFrame(
        data = {
            "Team": results.keys(), 
            "AUC": [x[0] for x in results.values()],
            "Time": [x[1] for x in results.values()],
        }
    )
    print(prediction_frame)
    prediction_frame.to_excel(writer, index=False)
    writer.close()