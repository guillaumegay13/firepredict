import pandas as pd

from firepredict.evaluate import scores


def test_perfect_and_disjoint_predictions():
    true = pd.Series([0, 1, 0, 3, 0])
    assert scores(true, true)["iou"] == 1 and scores(true, true)["f1"] == 1
    s = scores(true, pd.Series([2, 0, 2, 0, 2]))
    assert s["iou"] == 0 and s["f1"] == 0 and s["recall"] == 0
