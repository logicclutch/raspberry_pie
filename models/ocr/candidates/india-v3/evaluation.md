# OCR candidate evaluation: PASS

- baseline: `/Users/ibm/Desktop/raspberry-main/models/ocr/plate_ocr.onnx`
- candidate: `/Users/ibm/Desktop/raspberry-main/models/ocr/candidates/india-v3/plate_ocr.onnx`
- config: `/Users/ibm/Desktop/raspberry-main/config.yaml`
- truth: `/Users/ibm/Desktop/raspberry-main/training/ground_truth/gate_cam.csv, /Users/ibm/Desktop/raspberry-main/training/ground_truth/phone.csv`
- crops: `/Users/ibm/Desktop/raspberry-main/training/runs/india-v3/splits/test.csv`

## Videos (real engine, voting on)

| | baseline | candidate |
|---|---|---|
| sure plates | 24 | 24 |
| correct | 21 | 23 |
| wrong | 1 | 0 |
| unscored (unsure vehicles) | 1 | 1 |
| missed | 3 | 1 |
| precision | 0.955 | 1.000 |
| recall | 0.875 | 0.958 |

Baseline confusions: 8>3 x1, 7>9 x1, M>E x1, 6>7 x1, 8>9 x1, 2>3 x1
Candidate confusions: none

- baseline WRONG vlc-record-2026-09-28-14h52m53s-rtsp___192.168.5.53_554_avstream_channel=1_stream=1.sdp-.mp4: truth HR87M6812, shown HR39E7913
- baseline unscored vlc-record-2026-09-28-14h52m53s-rtsp___192.168.5.53_554_avstream_channel=1_stream=1.sdp-.mp4: shown RJ14UN8156 (check by eye)
- baseline missed: RJ02CG6293 RJ40GA7317 RJ52GA6273
- candidate unscored vlc-record-2026-09-28-14h52m53s-rtsp___192.168.5.53_554_avstream_channel=1_stream=1.sdp-.mp4: shown RJ14UN8156 (check by eye)
- candidate missed: RJ40GA7317

## Crops (OCR on labelled crops, no voting)

| | baseline | candidate |
|---|---|---|
| crops | 216 | 216 |
| raw exact | 0.671 | 0.917 |
| validated exact | 0.667 | 0.898 |

Baseline confusions: 0>O x19, 4>- x11, G>- x10, 1>- x8, N>- x8, 2>- x5, 9>- x4, 6>- x4, 7>- x3, 0>- x3, G>S x3, H>- x3
Candidate confusions: G>- x10, N>- x10, ->9 x10, R>- x2, 0>9 x2, 1>- x1, C>7 x1, 1>7 x1, 5>3 x1, 4>9 x1, 2>H x1, 6>4 x1

## Gate

    OK   video wrong plates: candidate 0 <= baseline 1
    OK   video correct plates: candidate 23 >= baseline 21
    OK   video precision: candidate 1.0000 >= 0.0
    OK   no new plates on unsure vehicles
    OK   crop test set size: 216 >= 100
    OK   crop validated accuracy: candidate 0.8981 >= baseline 0.6667
    OK   candidate strictly better somewhere (more correct, fewer wrong, or higher crop accuracy)
