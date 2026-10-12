# OCR candidate evaluation: FAIL

- baseline: `/Users/ibm/Desktop/raspberry-main/models/ocr/plate_ocr.onnx`
- candidate: `/Users/ibm/Desktop/raspberry-main/models/ocr/candidates/india-v5/plate_ocr.onnx`
- config: `/Users/ibm/Desktop/raspberry-main/config.yaml`
- truth: `/Users/ibm/Desktop/raspberry-main/training/ground_truth/gate_cam.csv, /Users/ibm/Desktop/raspberry-main/training/ground_truth/phone.csv`
- crops: `(none)`

## Videos (real engine, voting on)

| | baseline | candidate |
|---|---|---|
| sure plates | 24 | 24 |
| correct | 23 | 22 |
| wrong | 0 | 0 |
| unscored (unsure vehicles) | 2 | 2 |
| missed | 1 | 2 |
| precision | 1.000 | 1.000 |
| recall | 0.958 | 0.917 |

Baseline confusions: none
Candidate confusions: none

- baseline unscored vlc-record-2026-09-28-14h52m53s-rtsp___192.168.5.53_554_avstream_channel=1_stream=1.sdp-.mp4: shown HR63D8065 (check by eye)
- baseline unscored vlc-record-2026-09-28-14h52m53s-rtsp___192.168.5.53_554_avstream_channel=1_stream=1.sdp-.mp4: shown RJ14UN8156 (check by eye)
- baseline missed: RJ02CG6293
- candidate unscored vlc-record-2026-09-28-14h52m53s-rtsp___192.168.5.53_554_avstream_channel=1_stream=1.sdp-.mp4: shown HR63D8065 (check by eye)
- candidate unscored vlc-record-2026-09-28-14h52m53s-rtsp___192.168.5.53_554_avstream_channel=1_stream=1.sdp-.mp4: shown RJ14UN8156 (check by eye)
- candidate missed: RJ40GA7317 RJ52GA6273

## Gate

    OK   video wrong plates: candidate 0 <= baseline 0
    FAIL video correct plates: candidate 22 >= baseline 23
    OK   video precision: candidate 1.0000 >= 0.0
    OK   no new plates on unsure vehicles
    FAIL crop test ran (no labelled held-out crops: label the test videos' crops, or --crops)
    FAIL candidate strictly better somewhere (more correct, fewer wrong, or higher crop accuracy)
