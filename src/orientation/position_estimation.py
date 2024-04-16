class PositionEstimation:
    def __init__(self, F: float, W: float, L: float):
        self.F = F  # focus of camera mm
        self.W = W  # width of camera mm
        self.L = L  # length of camera mm
        self.d = 0.04 # diameter of ball

    def calculate(self, w, l, x, y, cx, cy, d_pix):
        """
        (w,l) width and and lenght of the image
        (x,y) x and y coordinates of the center of the ping pong ball
        (cx,cy) center of the image 
        d_pix pixels of diameter of ping pong ball
        """
        fx = self.F * self.W / w
        fy = self.F * self.L / l
        zreal = (fx * self.d) / d_pix
        xreal = ((x-cx) * zreal) / fx
        yreal = ((y-cy) * zreal) / fy

        return (xreal, yreal, zreal)

