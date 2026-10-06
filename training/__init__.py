"""Dev-machine tools for building the Indian-plate OCR fine-tuning dataset (never deployed to the Pi).

harvest  -> replay videos through the real pipeline and save plate crops + OCR suggestions
labeler  -> local web app where a human types / confirms the plate text
split    -> group verified crops by vehicle and write train/val/test CSVs for the trainer
synth    -> optional synthetic Indian plates (a supplement, never the test set)
train.sh -> one command: split -> fine-tune -> ONNX -> models/ocr/candidates/<run>/ -> evaluate
evaluate -> gate: candidate vs the live OCR model on the ground-truth videos + held-out crops
"""
