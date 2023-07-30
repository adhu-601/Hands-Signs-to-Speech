import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def gather_data(num_samples):
     
    global rock, paper, scissor, nothing
     
    # Initialize the camera
    cap = cv2.VideoCapture(0)
 
    # trigger tells us when to start recording
    trigger = False
     
    # Counter keeps count of the number of samples collected
    counter = 0
     
    # This the ROI size, the size of images saved will be box_size -10
    box_size = 234
     
    # Getting the width of the frame from the camera properties
    width = int(cap.get(3))
 
 
    while True:
         
        # Read frame by frame
        ret, frame = cap.read()
         
        # Flip the frame laterally
        frame = cv2.flip(frame, 1)
         
        # Break the loop if there is trouble reading the frame.
        if not ret:
            break
             
        # If counter is equal to the number samples then reset triger and the counter
        if counter == num_samples:
            trigger = not trigger
            counter = 0
         
        # Define ROI for capturing samples
        cv2.rectangle(frame, (width - box_size, 0), (width, box_size), (0, 250, 150), 2)
         
        # Make a resizable window.
        cv2.namedWindow("Collecting images", cv2.WINDOW_NORMAL)
         
         
        # If trigger is True than start capturing the samples
        if trigger:
             
            # Grab only slected roi
            roi = frame[5: box_size-5 , width-box_size + 5: width -5]
             
            # Append the roi and class name to the list with the selected class_name
            eval(class_name).append([roi, class_name])
                                     
            # Increment the counter 
            counter += 1
         
            # Text for the counter
            text = "Collected Samples of {}: {}".format(class_name, counter)
             
        else:
            text = "Press 'r' to collect rock samples, 'p' for paper, 's' for scissor and 'n' for nothing"
         
        # Show the counter on the imaege
        cv2.putText(frame, text, (3, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
         
        # Display the window
        cv2.imshow("Collecting images", frame)
         
        # Wait 1 ms
        k = cv2.waitKey(1)
         
        # If user press 'r' than set the path for rock directoryq
        if k == ord('r'):
             
            # Trigger the variable inorder to capture the samples
            trigger = not trigger
            class_name = 'rock'
            rock = []
            
             
        # If user press 'p' then class_name is set to paper and trigger set to True  
        if k == ord('p'):
            trigger = not trigger
            class_name = 'paper'
            paper = []
         
        # If user press 's' then class_name is set to scissor and trigger set to True  
        if k == ord('s'):
            trigger = not trigger
            class_name = 'scissor'
            scissor = []
                     
        # If user press 's' then class_name is set to nothing and trigger set to True
        if k == ord('n'):
            trigger = not trigger
            class_name = 'nothing'
            nothing = []
         
        # Exit if user presses 'q'
        if k == ord('q'):
            break
             
    #  Release the camera and destroy the window
    cap.release()
    cv2.destroyAllWindows()

no_of_samples = 100
gather_data(no_of_samples)

plt.figure(figsize=[30,20])
 
# Set the rows and columns
rows, cols = 4, 8
 
# Iterate for each class
for class_index, each_list in enumerate([rock, paper, scissor,nothing]):
     
    # Get 8 random indexes, since we will be showing 8 examples of each class.
    r = np.random.randint(no_of_samples, size=8);
     
    # Plot the examples
    for i, example_index in enumerate(r,1):
        plt.subplot(rows,cols,class_index*cols + i );plt.imshow(each_list[example_index][0][:,:,::-1]);plt.axis('off');
