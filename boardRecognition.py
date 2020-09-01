'''This file processes the image of the sudoku puzzle.
    It applies image preprocessing, determines the grid, adjusts the perspective, and extracts each tile.
    Finally, it determines which tiles are empty, constructs the puzzle skeleton, and saves all number tile images to NumberCells folder.
'''

import cv2
import numpy as np
import os


def preprocess_image(image):
    ''' 
    Description:
    Function that reads the image, applies image processing, determines the grid, adjusts the perspective, and extracts each tile. 

    Parameters:
    image: numpy.ndarray of the original board image

    Returns: numpy.ndarray of the processed board image
    '''
    def perspective_transform(image, corners):
        ''' Helper function to crop image of sudoku board to fit the puzzle'''
        def order_corner_points(corners):
            # Separate corners into points
            corners = [(corner[0][0], corner[0][1]) for corner in corners]
            # 0 = top right, 1 = top left, 2 = bottom left, 3 = bottom right
            top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
            return (top_l, top_r, bottom_r, bottom_l)

        # Order the points clockwise
        ordered_corners = order_corner_points(corners)
        top_l, top_r, bottom_r, bottom_l = ordered_corners

        # Determine width of new image which is the max distance between
        # (bottom right and bottom left) or (top right and top left) x-coordinates
        width_A = np.sqrt(((bottom_r[0] - bottom_l[0])
                           ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
        width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) +
                          ((top_r[1] - top_l[1]) ** 2))
        width = max(int(width_A), int(width_B))

        # Determine height of new image which is the max distance between
        # (top right and bottom right) or (top left and bottom left) y-coordinates
        height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) +
                           ((top_r[1] - bottom_r[1]) ** 2))
        height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) +
                           ((top_l[1] - bottom_l[1]) ** 2))
        height = max(int(height_A), int(height_B))

        # Construct new points to obtain top-down view of image in
        # top_r, top_l, bottom_l, bottom_r order
        dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                               [0, height - 1]], dtype="float32")

        # Convert to Numpy
        ordered_corners = np.array(ordered_corners, dtype="float32")

        # Determine the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

        # Return final image
        return cv2.warpPerspective(image, matrix, (width, height))

    # Read original image of puzzle
    img = image
    cv2.imwrite('./Steps/1.png', img)
    original = img.copy()

    # Make the image greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./Steps/2.png', gray)

    # Blur the image
    blur = cv2.medianBlur(gray, 3)
    cv2.imwrite('./Steps/3.png', blur)

    # Apply adaptive threshold to the image
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    cv2.imwrite('./Steps/4.png', blur)

    # Crop the image to fit puzzle size
    conts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    conts = conts[0] if len(conts) == 2 else conts[1]
    conts = sorted(conts, key=cv2.contourArea, reverse=True)

    for c in conts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        transformed = perspective_transform(original, approx)
        break

    # Rotate the image
    rotated = cv2.rotate(transformed, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite('./Steps/5.png', rotated)

    # Apply greyscale to the image
    board = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./Steps/6.png', board)

    # Invert the image
    board = cv2.bitwise_not(board)
    cv2.imwrite('./Steps/7.png', board)

    # Apply adaptive threshold
    board = cv2.adaptiveThreshold(
        board, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 1111, 1)
    cv2.imwrite('./Steps/8.png', board)

    return board


def extract_tiles(board):
    ''' 
    Description:
    Function that extracts an image of each individual tile from the processed puzzle, stores them in a numpy array, and saves them to BoardCells folder 

    Parameters:
    board: numpy.ndarray of the processed board image

    Returns:
    numpy array of individual tile images
    '''
    # Create a 9x9 numpy array of the individual cell images
    grid = np.copy(board)
    edge = np.shape(grid)[0]
    celledge = edge // 9

    tempgrid = []
    for i in range(celledge, edge+1, celledge):
        for j in range(celledge, edge+1, celledge):
            rows = grid[i-celledge:i]
            tempgrid.append([rows[k][j-celledge:j] for k in range(len(rows))])

    finalgrid = []
    for i in range(0, len(tempgrid)-8, 9):
        finalgrid.append(tempgrid[i:i+9])

    for i in range(9):
        for j in range(9):
            finalgrid[i][j] = np.array(finalgrid[i][j])

    # Process all cell images as 28x28 pixels

    def pixelate(cell):
        cell = cv2.resize(cell, (28, 28))
        return cell

    for i in range(9):
        for j in range(9):
            finalgrid[i][j] = pixelate(finalgrid[i][j])

    # Write all 9x9 processed cell images
    for i in range(9):
        for j in range(9):
            cv2.imwrite(str("BoardCells/cell"+str(i) +
                            str(j)+".jpg"), finalgrid[i][j])

    def scaling(img, size, margin=20, background=0):
        '''Helper function for scaling and centering the image '''
        h, w = img.shape[:2]

        def center_length(length):
            '''Centering for some length'''
            if length % 2 == 0:
                side_1 = int((size - length) / 2)
                side_2 = side_1
            else:
                side_1 = int((size - length) / 2)
                side_2 = side_1 + 1
            return side_1, side_2

        def scale(r, x):
            return int(r * x)

        if h > w:
            top_pad = int(margin / 2)
            bottom_pad = top_pad
            ratio = (size - margin) / h
            w, h = scale(ratio, w), scale(ratio, h)
            left_pad, right_pad = center_length(w)
        else:
            left_pad = int(margin / 2)
            right_pad = left_pad
            ratio = (size - margin) / w
            w, h = scale(ratio, w), scale(ratio, h)
            top_pad, bottom_pad = center_length(h)

        img = cv2.resize(img, (w, h))
        img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad,
                                 right_pad, cv2.BORDER_CONSTANT, None, background)
        return cv2.resize(img, (size, size))

    def extractNum(img):
        '''Helper function to extract number from tile image'''
        extracted = img
        thresh = 128  # define a threshold, 128 is the middle of black and white in grey scale
        gray = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
        # Finding contours
        conts = cv2.findContours(
            gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        conts = conts[0] if len(conts) == 2 else conts[1]
        for c in conts:
            x, y, w, h = cv2.boundingRect(c)
            if (x < 3 or y < 3 or h < 3 or w < 3):
                continue
            extracted = gray[y:y + h, x:x + w]
            extracted = scaling(extracted, 120)
        return extracted

    # Extract numbers from tile images
    for i in range(9):
        for j in range(9):
            finalgrid[i][j] = extractNum(finalgrid[i][j])

    return finalgrid


def organize_data(finalgrid):
    '''
    Description:
    Function that determines which tiles are empty, constructs a skeleton of the puzzle based on this information, and saves this in Puzzle.txt 
    Filters tile images to the images with numbers, and saves these images in NumberCells folder to be recognized later

    Parameters:
    finalgrid: numpy array of tile images

    Returns:
    None
    '''
    puzzle = [['?' for i in range(9)] for j in range(9)]
    # Identify blank tiles based on image size
    # 0 = blank
    # ? = not yet identified
    for i in range(9):
        for j in range(9):
            if finalgrid[i][j].shape[0] == 28:
                # Tile is blank -> mark it with zero
                puzzle[i][j] = 0
            else:
                # Indicate current tile is a number rather than a blank (although it is unknown)
                puzzle[i][j] = '?'
                # Write number image to a file to be recognized later
                img = cv2.resize(finalgrid[i][j], (140, 140))
                cv2.imwrite(str("NumberCells/cell"+str(i) +
                                str(j)+".jpg"), img)

    for row in puzzle:
        print(row)

    # Write puzzle skeleton with blanks to txt file
    puzzle_file = open("Puzzle.txt", "w")
    for i in range(9):
        for j in range(9):
            puzzle_file.write(str(puzzle[i][j]))
    puzzle_file.close()


# Preprocess image of sudoku puzzle
board = preprocess_image(cv2.imread('example.png'))
# Extract tiles from the sudoku puzzle
finalgrid = extract_tiles(board)
# Determine empty tiles, and save puzzle skeleton and number images
organize_data(finalgrid)
