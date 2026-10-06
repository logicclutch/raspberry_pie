# OCR candidate evaluation: FAIL

- baseline: `/Users/ibm/Desktop/raspberry-main/models/ocr/plate_ocr.onnx`
- candidate: `/Users/ibm/Desktop/raspberry-main/models/ocr/candidates/kag2line-v1/plate_ocr.onnx`
- config: `/Users/ibm/Desktop/raspberry-main/config.yaml`
- truth: `/Users/ibm/Desktop/raspberry-main/training/ground_truth/gate_cam.csv, /Users/ibm/Desktop/raspberry-main/training/ground_truth/phone.csv`
- crops: `(none)`

## Videos (real engine, voting on)

| | baseline | candidate |
|---|---|---|
| sure plates | 23 | 23 |
| correct | 21 | 16 |
| wrong | 1 | 2 |
| unscored (unsure vehicles) | 1 | 0 |
| missed | 2 | 7 |
| precision | 0.955 | 0.889 |
| recall | 0.913 | 0.696 |

Baseline confusions: 8>3 x1, 7>9 x1, M>E x1, 6>7 x1, 8>9 x1, 2>3 x1
Candidate confusions: 8>3 x1, 7>9 x1, M>E x1, 6>7 x1, 8>9 x1, 2>3 x1, Q>0 x1, 6>- x1

- baseline WRONG vlc-record-2026-09-28-14h52m53s-rtsp___192.168.5.53_554_avstream_channel=1_stream=1.sdp-.mp4: truth HR87M6812, shown HR39E7913
- baseline unscored vlc-record-2026-09-28-14h52m53s-rtsp___192.168.5.53_554_avstream_channel=1_stream=1.sdp-.mp4: shown RJ14UN8156 (check by eye)
- baseline missed: RJ02CG6293 RJ40GA7317
- candidate WRONG vlc-record-2026-09-28-14h52m53s-rtsp___192.168.5.53_554_avstream_channel=1_stream=1.sdp-.mp4: truth HR87M6812, shown HR39E7913
- candidate WRONG IMG_1565.mp4: truth UP16DQ9256, shown UP16D0925
- candidate missed: GJ12BX3886 HR47F1710 NL01AC0056 RJ02CG6293 RJ40GA2997 RJ40GA7317 UP16DQ9256

## Gate

    FAIL video wrong plates: candidate 2 <= baseline 1
    FAIL video correct plates: candidate 16 >= baseline 21
    OK   video precision: candidate 0.8889 >= 0.0
    OK   no new plates on unsure vehicles
    FAIL crop test ran (no labelled held-out crops: label the test videos' crops, or --crops)
    FAIL candidate strictly better somewhere (more correct, fewer wrong, or higher crop accuracy)
