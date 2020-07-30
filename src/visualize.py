import cv2

class Visualizer:
    def __init__(self, frame, fd_img, fd_coords, ld_coords, hpe_val):
        self.frame = frame
        self.fd_img = fd_img
        self.fd_coords = fd_coords
        self.ld_coords = ld_coords
        self.hpe_val = hpe_val
        
    def visualize(self):
        if self.fd_coords == []:
            cv2.putText(self.frame, 'No face detected', (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
        else:
            cv2.rectangle(self.fd_img, (self.fd_coords[0], self.fd_coords[1]), (self.fd_coords[2], self.fd_coords[3]), (0, 0, 255), 1)
            cv2.rectangle(self.fd_img, (self.ld_coords[0]), (self.ld_coords[1]), (0, 0, 255), 1)
            cv2.rectangle(self.fd_img, (self.ld_coords[2]), (self.ld_coords[3]), (0, 0, 255), 1)
            cv2.putText(self.fd_img, 'Yaw: {:2f}'.format(self.hpe_val[0]) , (2, 15), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(self.fd_img, 'Pitch: {:2f}'.format(self.hpe_val[1]) , (2, 30), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(self.fd_img, 'Roll: {:2f}'.format(self.hpe_val[2]) , (2, 45), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imshow('Frame', cv2.resize(self.fd_img, (400, 400)))
