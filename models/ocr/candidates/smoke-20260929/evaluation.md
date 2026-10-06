# OCR candidate evaluation: FAIL

- baseline: `/Users/ibm/Desktop/raspberry-main/models/ocr/plate_ocr.onnx`
- candidate: `/Users/ibm/Desktop/raspberry-main/models/ocr/candidates/smoke-20260929/plate_ocr.onnx`
- config: `/Users/ibm/Desktop/raspberry-main/config.yaml`
- truth: `/Users/ibm/Desktop/raspberry-main/training/ground_truth/gate_cam.csv, /Users/ibm/Desktop/raspberry-main/training/ground_truth/phone.csv`
- crops: `/Users/ibm/Desktop/raspberry-main/training/runs/smoke-20260929/splits/test.csv`

## Videos (real engine, voting on)

| | baseline | candidate |
|---|---|---|
| sure plates | 23 | 23 |
| correct | 19 | 21 |
| wrong | 0 | 0 |
| unscored (unsure vehicles) | 0 | 0 |
| missed | 4 | 2 |
| precision | 1.000 | 1.000 |
| recall | 0.826 | 0.913 |

Baseline confusions: none
Candidate confusions: none

- baseline missed: RJ02CG6293 RJ30CD0303 RJ40GA2997 RJ40GA7317
- candidate missed: RJ30CD0303 RJ40GA7317

## Crops (OCR on labelled crops, no voting)

| | baseline | candidate |
|---|---|---|
| crops | 72 | 72 |
| raw exact | 0.528 | 0.667 |
| validated exact | 0.486 | 0.597 |

Baseline confusions: 3>- x9, 7>- x7, 0>- x7, 1>- x6, A>- x5, 0>O x5, 8>- x5, 4>- x3, 4>A x3, G>- x2, 3>7 x2, R>- x2
Candidate confusions: 4>A x6, 2>- x4, 0>- x3, 3>7 x2, A>7 x2, 3>- x2, C>E x2, 3>9 x2, 6>5 x2, 2>Z x2, A>G x1, 1>3 x1

## Gate

    FAIL no leak: candidate may have been trained on the test data: IMG_1565.mp4
    OK   video wrong plates: candidate 0 <= baseline 0
    OK   video correct plates: candidate 21 >= baseline 19
    OK   video precision: candidate 1.0000 >= 0.0
    OK   no new plates on unsure vehicles
    FAIL crop test set size: 72 >= 100
    OK   crop validated accuracy: candidate 0.5972 >= baseline 0.4861
    OK   candidate strictly better somewhere (more correct, fewer wrong, or higher crop accuracy)
