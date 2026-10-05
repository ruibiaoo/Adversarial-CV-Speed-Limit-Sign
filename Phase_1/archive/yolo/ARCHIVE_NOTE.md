Archived YOLO classification baseline.

This was Phase 1 before the tensorflow migration.
Model was yolo26-cls (yolo26m-cls.pt / yolo26n-cls.pt).
train.py + modules/ + config.yaml + tuner.yaml are the old training code.
runs/ has the old trained weights and plots.
tuning/summaries/ has the old Ray Tune best hyperparameters.
tuning/classify/ and tuning/detect/ (if present) are just old Ray trial
outputs, they can be deleted to save disk. They are gitignored.

Kept so Phase 2 can still compare attacks against YOLO.
New tensorflow code lives in Phase_1/ (outside this folder).
