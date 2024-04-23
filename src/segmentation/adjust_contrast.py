import cv2

def adjust_contrast(image, clip_limit=2.0, tile_grid_size=(8,8)):

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Split into the L, A, and B components
    l, a, b = cv2.split(lab)
    # Apply CLAHE to the L component
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    # Merge the CLAHE enhanced L component back with A and B
    limg = cv2.merge((cl,a,b))
    # Convert back to RGB
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final