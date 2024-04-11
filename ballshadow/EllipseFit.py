import cv2
import numpy as np
import matplotlib.pyplot as plt

class FindOriginalBall:
    # make a better name lol
    # test this with images with more noise in it
    def __init__(self, img):
        self.img = cv2.imread(img)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def _find_ellipses(self, threshold = 0):
        
        # Threshold the image set between 0(high tolerance) and 255(low tolerance) 255 makes it harder to find elipses
        ret, thresh = cv2.threshold(self.gray, threshold, 255, 0)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if contours are found
        if len(contours) != 0:
            # Iterate through contours
            for cont in contours:
                if len(cont) < 5:
                    break
                # Fit ellipse to contour
                ellipse = cv2.fitEllipse(cont)
                return ellipse  # Return the first fitted ellipse
        return None


    def _create_ellipse(self,img, ellipse):
        # Draw ellipse on image
        img_with_ellipse = cv2.ellipse(img.copy(), ellipse, (0, 255, 0), 2)
        
        # Get center coordinates of the ellipse
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        
        # Draw black dot at the center
        cv2.circle(img_with_ellipse, center, 5, (0, 0, 0), -1)  # -1 to fill the circle
        
        return img_with_ellipse
    

    def _generate_major_axis_points(self, ellipse, num_points=100):
        # Unpack ellipse parameters
        center, (semi_major, semi_minor), angle = ellipse
        
        # Convert angle to radians
        angle_rad = np.deg2rad(90-angle)
        
        # Calculate endpoints of major axis
        major_axis_end1 = (center[0] + semi_major * np.cos(angle_rad), center[1] - semi_major * np.sin(angle_rad))
        major_axis_end2 = (center[0] - semi_major * np.cos(angle_rad), center[1] + semi_major * np.sin(angle_rad))
        
        # Generate points along the major axis
        t = np.linspace(0, 1, num_points)
        major_axis_points = (1 - t[:, None]) * major_axis_end1 + t[:, None] * major_axis_end2
        
        return major_axis_points
    

    def _valid_point(self,ellipse, point1, point2):
        major_axis_points = self._generate_major_axis_points(ellipse)
        p1 = np.array(point1)
        p2 = np.array(point2)
        
        # Calculate distances from each point to major axis
        dist1 = np.min(np.linalg.norm(major_axis_points-p1, axis=1))
        dist2 = np.min(np.linalg.norm(major_axis_points-p2, axis=1))

        
        # Determine which point is closer to major axis
        if dist1 < dist2:
            return point1[1]
        else:
            return point2[1]
        

    def graph_to_original(self, xdir=1):

        # Find ellipse
        ellipse = self._find_ellipses()
        circle = self.calculate_original(ellipse,xdir)
        circleellipse = (circle[0],(circle[1]*2,circle[1]*2),0)
        # Draw ellipse on original image
        if ellipse is not None:
            img1 = self._create_ellipse(self.img,ellipse)
            fin_image = self._create_ellipse(img1, circleellipse)

            # Convert BGR image to RGB (matplotlib uses RGB)
            fin_image_rgb = cv2.cvtColor(fin_image, cv2.COLOR_BGR2RGB)

            # Display the image with ellipse using matplotlib
            plt.imshow(fin_image_rgb)

            plt.axis('off')  # Turn off axis
            plt.show()
        else:
            print("No ellipse found.")
    

    def calculate_original(self, ellipse = None, xdir=1):
        if not ellipse:
            ellipse = self._find_ellipses()
        if ellipse:
            x1, y1 = ellipse[0]
            alpha = 90 - ellipse[2] # switch to x axis
            s,l = ellipse[1]
            x2 = x1 + (l/2-s/2)*np.cos(np.radians(alpha)) * xdir
            y2_1 = y1 + (l/2-s/2)*np.sin(np.radians(alpha)) * xdir
            y2_2 = y1 + (l/2-s/2)*np.sin(np.radians(alpha)) * -xdir
            y2 = self._valid_point(ellipse,[x2,y2_1],[x2,y2_2])

            # equations were found on paper 5
            return ((int(x2),int(y2)),int(s/2))
        return (0,0),0

