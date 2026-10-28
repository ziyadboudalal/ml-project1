"""Reproduce Ismail's submission from the saved model, without retraining."""

from submissions.ismail import ROOT, reproduce


if __name__ == "__main__":
    reproduce(ROOT / "dataset", ROOT / "submissions" / "ismail")
