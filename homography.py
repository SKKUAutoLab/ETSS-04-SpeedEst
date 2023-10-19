import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_homography_matrix(source, destination):
    """ Calculates the entries of the Homography matrix between two sets of matching points.

    Args
    ----
        - `source`: Source points where each point is int (x, y) format.
        - `destination`: Destination points where each point is int (x, y) format.

    Returns
    ----
        - A numpy array of shape (3, 3) representing the Homography matrix.

    Raises
    ----
        - `source` and `destination` is lew than four points.
        - `source` and `destination` is of different size.
    """
    assert len(source) >= 4, "must provide more than 4 source points"
    assert len(destination) >= 4, "must provide more than 4 destination points"
    assert len(source) == len(destination), "source and destination must be of equal length"
    A = []
    b = []
    for i in range(len(source)):
        s_x, s_y = source[i]
        d_x, d_y = destination[i]
        A.append([s_x, s_y, 1, 0, 0, 0, (-d_x)*(s_x), (-d_x)*(s_y)])
        A.append([0, 0, 0, s_x, s_y, 1, (-d_y)*(s_x), (-d_y)*(s_y)])
        b += [d_x, d_y]
    A = np.array(A)
    h = np.linalg.lstsq(A, b)[0]
    h = np.concatenate((h, [1]), axis=-1)
    # print('This is A: ', A)
    # print('This is b: ', b)
    # print('This is homorgraphy metrix: ', h)
    return np.reshape(h, (3, 3))



if __name__ == "__main__":
    a = 0.3
    source_points = np.array([
        [337.61311911, 237.74363066],
        [834.39104709, 190.72425485],
        [870.39184488, 807.86437673],
        [1777.91043, 689.098578],
        [1.41175153e+03, 4.94171479e+02],
    ])
    destination_points = np.array([
        [0, 0],
        [7.51/a, 0],
        [0, 27.865/a],
        [7.492/a, 28.077/a],
        [7.51/a, 21.042/a]
    ])

    # source_points = np.array([
    #     [1, -1],
    #     [-1, 1],
    #     [-2, 2],
    #     [2, -2],
    # ])
    # destination_points = np.array([
    #     [3, 0],
    #     [1, 4],
    #     [0, 6],
    #     [4,-2]
    # ])
    # #  28.077
    # source_image = cv2.imread("sample.jpeg")
    # t_source_image = source_image.copy()

    # # draw markings on the source image
    # for i, pts in enumerate(source_points):
    #     cv2.putText(source_image, str(i+1), (pts[0] + 15, pts[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 215, 255), 5)
    #     cv2.circle(source_image, pts, 10, (0, 215, 255), 10)

    h = get_homography_matrix(source_points, destination_points)
    H = cv2.findHomography(source_points, destination_points)
    # destination_image = cv2.warpPerspective(t_source_image, h, (300, 300))
    #
    # figure = plt.figure(figsize=(12, 6))
    #
    # subplot1 = figure.add_subplot(1, 2, 1)
    # subplot1.title.set_text("Source Image")
    # subplot1.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    #
    # subplot2 = figure.add_subplot(1, 2, 2)
    # subplot2.title.set_text("Destination Image")
    # subplot2.imshow(cv2.cvtColor(destination_image, cv2.COLOR_BGR2RGB))

    # plt.show()
    # plt.savefig("output.png")

    # #DEBUG
    # # Create a column of ones with the same number of rows as the original array
    # ones_column = np.ones((source_points.shape[0], 1), dtype=source_points.dtype)
    # # Extend the original array with the ones column
    # new_source = np.column_stack((source_points, ones_column))
    # new_destination = np.column_stack((destination_points, ones_column))
    # print('This is new_destination metrix: ', new_destination.T)
    #
    # H = np.dot(new_destination.T, np.linalg.inv(new_source.T))

    # Son add print
    print('This is homorgraphy metrix: ', h)
    print('This is homorgraphy metrix  with opencv: ', H)
