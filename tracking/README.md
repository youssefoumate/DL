# Current Progress
## Sampling functions (Generate bbox candidates)

- [x] using normal distribution

- [x] using kalman filter (used partially)

## Tracking

- [x] Train the backbone using a bbox regressor to recognize the target.

- [ ] Train a bbox classifier to detect the target using samples from the backbone output.

# Dependencies

- pytorch 2.0
- opencv-python
- numpy
- tqdm

# Usage
- python3 tracker.py
