import cv2

def win_imshow(label, img):
    """Before launching cv2.imshow(), set a mouse event callback.
        When the user left-clicks on the image, print out the pixel
        position and pixel value.

        Note: The callback show_pixel_info will capture the value of
        label and img.

    """
    
    def show_pixel_info(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel_value = img[y, x]
            print(f'{label} ({x}, {y}): {pixel_value}')
            
    cv2.namedWindow(label)
    cv2.setMouseCallback(label, show_pixel_info)
    cv2.imshow(label, img)
