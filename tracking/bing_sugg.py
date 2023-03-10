import cv2
import numpy as np

# Number of points to track
N = 10

# Create random initial points
points = np.random.randint(0, 1000, (N, 2))

# Create a Kalman filter object for each point
kfs = []
for i in range(N):
    kf = cv2.KalmanFilter(4, 2) # State size: 4 (x, y, dx, dy), Measurement size: 2 (x, y)
    kf.transitionMatrix = np.array([[1., 0., 1., 0.], # State transition model
                                    [0., 1., 0., 1.],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
    kf.transitionMatrix = kf.transitionMatrix.astype(np.float32)
    kf.measurementMatrix = np.array([[1., 0., 0., 0.], # Measurement model
                                     [0., 1., 0., 0.]]).astype(np.float32)
    kf.processNoiseCov = np.eye(4).astype(np.float32) * (10 ** -3) # Process noise covariance
    kf.measurementNoiseCov = np.eye(2).astype(np.float32) * (10 ** -2) # Measurement noise covariance
    kf.errorCovPost = np.eye(4).astype(np.float32) * (10 ** -3) # Posteriori error estimate covariance matrix
    kf.statePost = np.array([points[i][0], points[i][1], 
                             np.random.randn(), 
                             np.random.randn()], dtype=np.float32).reshape(-1,1) # Initial state estimate
    
    kfs.append(kf)

# Create an image to draw on
img = np.zeros((1000,1000), dtype=np.uint8)

while True:
    print("something")
    # Simulate random measurements for each point with some noise
    measurements = []
    for i in range(N):
        if points[i][0] == 999 or points[i][0]==999:
            points[i] = np.random.randint(0, 1000, (1, 2))
        x,y = points[i]
        dx = int(np.random.normal(5)) # Random displacement along x-axis
        dy = int(np.random.normal(5)) # Random displacement along y-axis
        
        x += dx 
        y += dy
        
        x += int(np.random.normal(5)) # Add some measurement noise along x-axis
        y += int(np.random.normal(5)) # Add some measurement noise along y-axis
        
        x = max(min(x,999),0) # Keep x within bounds of image width 
        y = max(min(y,999),0) # Keep y within bounds of image height
        
        measurements.append([x,y])
        
        points[i] = [x,y] 
    
    measurements = np.array(measurements).astype(np.float32)
    print(x,y)
    
    img.fill(255) # Clear image
    
    for i in range(N):
        
        prediction = kfs[i].predict()[:2] # Predict next state of point i
        
        correction = kfs[i].correct(measurements[i].reshape(-1,1))[:2] # Correct state estimate with measurement of point i
        
        prediction_x,prediction_y= prediction.flatten().astype(int)
        
        correction_x ,correction_y= correction.flatten().astype(int)
        
        measured_x ,measured_y= measurements[i].astype(int)
        
        cv2.circle(img,(prediction_x,prediction_y),3,(255-i*25,i*25,i*25),10,cv2.LINE_AA)# Draw predicted point
        
        cv2.circle(img,(correction_x ,correction_y ),3,(i*25,i*25,i*25),10,cv2.LINE_AA)# Draw corrected point
        
        cv2.circle(img,(measured_x ,measured_y ),3,(i*25,i*25,i*25),10,cv2.LINE_AA)# Draw measured point

        cv2.line(img,(prediction_x,prediction_y),(correction_x ,correction_y ),(0,0,255),1,cv2.LINE_AA) # Draw line from predicted to corrected point
        
        cv2.line(img,(correction_x ,correction_y ),(measured_x ,measured_y ),(0,255,0),1,cv2.LINE_AA) # Draw line from corrected to measured point
    cv2.imshow("Kalman Filter",img) # Show image
    
    key = cv2.waitKey(30) & 0xFF # Wait for key press
    if key == 27: # If ESC is pressed, exit loop
        break

cv2.destroyAllWindows() # Destroy all windows