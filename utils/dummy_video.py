import cv2
import numpy as np

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