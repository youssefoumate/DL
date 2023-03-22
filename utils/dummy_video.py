import cv2
import numpy as np

# Create a blank black image
def create_dummy_video(num_frames = 50):
    image = np.zeros((500, 500, 3), np.uint8)

    # Define initial position and velocity of the disk
    x = 150 # x coordinate of the center
    y = 150 # y coordinate of the center
    r = 10 # radius of the disk
    vx = 5 # x component of velocity
    vy = -5 # y component of velocity

    # Define a loop to animate the disk movement
    for _ in range (num_frames):
        # Clear the previous image by filling it with black color
        image[:] = (0, 0, 0)

        # Draw a green circle at the current position
        cv2.circle(image, (x, y), r, (0, 255, 0), -1)

        # Display the image
        # cv2.imshow("Green Disk", image)

        # Update the position based on velocity
        x += vx 
        y += vy

        # Check for collision with boundaries and reverse direction if needed
        if x + r > image.shape[1] or x - r < 0: # right or left boundary 
            vx *= -1 # reverse x component of velocity 
        if y + r > image.shape[0] or y - r < 0: # bottom or top boundary 
            vy *= -1 # reverse y component of velocity 
        yield image

        

# Destroy all windows 
cv2.destroyAllWindows()