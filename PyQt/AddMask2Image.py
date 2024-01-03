import cv2
import numpy as np

# Load your main image
main_image = cv2.imread('main_image.jpg')

# Generate the mask using your segmentation model (replace this with your actual code)
# Assuming you have the mask as a NumPy array with the same shape as the main image
# mask = your_segmentation_model(main_image)

# Make sure the mask has the same number of channels as the main image (3 for RGB)
if len(mask.shape) == 2:
    mask = cv2.merge([mask] * 3)

# Overlay the mask on the main image
alpha = 0.5  # You can adjust the alpha value for transparency
result = cv2.addWeighted(main_image, 1 - alpha, mask, alpha, 0)

# Display the result
cv2.imshow('Segmentation Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result if needed
cv2.imwrite('result_image.jpg', result)
