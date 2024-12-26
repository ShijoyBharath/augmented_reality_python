import cv2
from matplotlib import pyplot

def main():
    print("AR Scene running...")

    img = cv2.imread('scene.png', 0)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # Find key points with ORB
    key_points = orb.detect(img, None)

    # Compute the descriptors with ORB
    key_points, desc = orb.compute(img, key_points)

    # Draw only keypoints location, not size & orientation
    new_img = cv2.drawKeypoints(img, key_points, None, color=(0, 255, 0), flags=0)
    pyplot.imshow(new_img)
    pyplot.show()


if __name__ == "__main__":
    main()
