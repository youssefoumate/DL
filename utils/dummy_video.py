import cv2
import numpy as np

def cvt_video2img(path):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(path)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    # Read until video is completed
    frame_id = 0
    while(cap.isOpened()):
        frame_id += 1
        if frame_id % 100 == 0:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                cv2.imwrite(f"input/{frame_id}.jpg", frame)
            else: 
                break
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()
    
# Create a blank black image
def create_dummy_video(num_frames = 50):

    # Define initial position and velocity of the disk
    x = [150, 200] # x coordinates of the centers
    y = [150, 200] # y coordinates of the centers
    r = [30] * len(x) # radii of the disks
    vx = [5] * len(x) # x components of velocity
    vy = [-5] * len(y) # y components of velocity

    # Define a loop to animate the disk movement
    for _ in range (num_frames):
        image = np.random.random((500, 500, 3)) * 255
        image = image.astype(dtype=np.uint8)
        # Draw green circles at the current positions
        for i in range(len(x)): # loop over the number of disks
            cv2.circle(image, (x[i], y[i]), r[i], (0, 0, 255), -1)

        # Display the image
        # cv2.imshow("Green Disks", image)

        # Update the positions based on velocities
        for i in range(len(x)): # loop over the number of disks
            x[i] += vx[i] 
            y[i] += vy[i]

            # Check for collision with boundaries and reverse direction if needed
            if x[i] + r[i] > image.shape[1] or x[i] - r[i] < 0: # right or left boundary 
                vx[i] *= -1 # reverse x component of velocity 
            if y[i] + r[i] > image.shape[0] or y[i] - r[i] < 0: # bottom or top boundary 
                vy[i] *= -1 # reverse y component of velocity 
        yield image, [x[0], y[0], r[0]*2, r[0]*2]

        

# Destroy all windows 
cv2.destroyAllWindows()