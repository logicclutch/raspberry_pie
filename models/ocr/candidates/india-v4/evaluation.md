# OCR candidate evaluation: FAIL

- baseline: `/Users/ibm/Desktop/raspberry-main/models/ocr/plate_ocr.onnx`
- candidate: `/Users/ibm/Desktop/raspberry-main/models/ocr/candidates/india-v4/plate_ocr.onnx`
- config: `/Users/ibm/Desktop/raspberry-main/config.yaml`
- truth: `/Users/ibm/Desktop/raspberry-main/training/ground_truth/gate_cam.csv, /Users/ibm/Desktop/raspberry-main/training/ground_truth/phone.csv`
- crops: `/Users/ibm/Desktop/raspberry-main/training/runs/india-v3/splits/test.csv`

## Videos (real engine, voting on)

| | baseline | candidate |
|---|---|---|
| sure plates | 24 | 24 |
| correct | 23 | 21 |
| wrong | 0 | 0 |
| unscored (unsure vehicles) | 1 | 1 |
| missed | 1 | 3 |
| precision | 1.000 | 1.000 |
| recall | 0.958 | 0.875 |

Baseline confusions: none
Candidate confusions: none

- baseline unscored vlc-record-2026-09-28-14h52m53s-rtsp___192.168.5.53_554_avstream_channel=1_stream=1.sdp-.mp4: shown RJ14UN8156 (check by eye)
- baseline missed: RJ40GA7317
- candidate unscored vlc-record-2026-09-28-14h52m53s-rtsp___192.168.5.53_554_avstream_channel=1_stream=1.sdp-.mp4: shown RJ14UN8156 (check by eye)
- candidate missed: RJ30CD0303 RJ40GA7317 RJ52GA6273

## Crops (OCR on labelled crops, no voting)

| | baseline | candidate |
|---|---|---|
| crops | 216 | 216 |
| raw exact | 0.917 | 0.921 |
| validated exact | 0.898 | 0.917 |

Baseline confusions: G>- x10, N>- x10, ->9 x10, R>- x2, 0>9 x2, 1>- x1, C>7 x1, 1>7 x1, 5>3 x1, 4>9 x1, 2>H x1, 6>4 x1
Candidate confusions: G>- x8, N>- x8, ->9 x8, G>M x2, N>2 x2, 9>3 x2, 2>9 x2, 3>9 x2, R>- x2, 6>0 x2, 0>9 x2, 1>- x1

## Gate

    OK   video wrong plates: candidate 0 <= baseline 0
    FAIL video correct plates: candidate 21 >= baseline 23
    OK   video precision: candidate 1.0000 >= 0.0
    OK   no new plates on unsure vehicles
    OK   crop test set size: 216 >= 100
    OK   crop validated accuracy: candidate 0.9167 >= baseline 0.8981
    OK   candidate strictly better somewhere (more correct, fewer wrong, or higher crop accuracy)
