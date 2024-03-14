import cv2

# Read the video file
video_path = 'pc.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

# Background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
people_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get the frame dimensions
    frame_h, frame_w = gray_frame.shape[:2]

    # Define the region of interest (ROI) in the middle part of the frame
    roi_w, roi_h = frame_w // 3, frame_h // 3
    roi_x = (frame_w - roi_w) // 2
    roi_y = (frame_h - roi_h) // 2
    roi = gray_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(roi)

    # Perform thresholding to obtain a binary image
    _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter contours based on area
        area = cv2.contourArea(contour)
        if area > 161000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x + roi_x, y + roi_y), (x + w + roi_x, y + h + roi_y), (0, 255, 0), 2)
            people_count += 1
    # Line in y axis, mid frame        
    cv2.line(frame, (roi_x + roi_w // 2, 0), (roi_x + roi_w // 2, frame_h), (255, 0, 0), 2)
    # Display frame with bounding boxes and people count
    cv2.putText(frame, f'People Count: {people_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('People Counting', frame)
    cv2.imshow('Subtracted', fg_mask)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
