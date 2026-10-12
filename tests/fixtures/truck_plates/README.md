# Indian truck plates, marked HSRP / non-HSRP

69 plate crops used by `tests/test_hsrp_trucks.py` to measure the HSRP check (`anpr/hsrp.py`).

- **Source:** "Truck LicensePlate Dataset" by Amritesh Tiwari, Kaggle
  (`amriteshtiwari20/truck-licenseplate-dataset`, version 1), MIT licence. The original has 83 truck photos
  with plate boxes (YOLO format) and no HSRP labels.
- **What was done:** each plate box was cut out with a margin (1/4 of the plate width left and right, 1/2 of
  its height above and below) and shrunk so the plate is at most 400 px wide. `box` in `labels.json`
  is the plate box inside the crop, in pixels (x1, y1, x2, y2).
- **Labels (by eye, from the photos):** `hsrp` = hologram and/or "IND" visible left of the number (13);
  `non_hsrp` = old painted/printed plate with no marks (56). Left out: 15 too blurred or cut off to tell,
  3 foreign plates (Bhutan, Israel), 2 boxes that are not a plate. A few `hsrp` labels rest on a small
  mark seen in a blurry photo (train_29, train_30, train_72).

## MIT licence of the source dataset

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions: The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED.
