''' This file reads data from a sudoku puzzle, solves the puzzle using backtracking, and prints the solved puzzle.
'''

import numpy as np

# Read puzzle data from txt file
puzzle_file = open("Puzzle.txt", 'r')
# Save the characters in the file in a list
board = list(puzzle_file.read())
puzzle_file.close()
# Convert characters in board list to integers
board = [int(i) for i in board]
# Resize 1D list to a 2D list of each row
board = np.reshape(board, (-1, 9))
print('\n Given Puzzle:')


def print_board(board):
    ''' Description:
    A function to print the puzzle 

    Parameters:
    board: 2d list of integers representing all the tiles in each row of the puzzle

    Returns:
    None
    '''
    print("")
    for i in range(len(board)):
        if i % 3 == 0 and i != 0:
            print("------|-------|------")
        for j in range(len(board[0])):
            if j % 3 == 0 and j != 0:
                print("| ", end="")

            if j == 8:
                if board[i][j] == 0:
                    print('.')
                else:
                    print(board[i][j])
            else:
                if board[i][j] == 0:
                    print('.' + " ", end="")
                else:
                    print(str(board[i][j]) + " ", end="")
    print("")


def is_legal(board, row, col, num):
    ''' Description:
    A function to determine if placing a given number in a given tile satisfies the sudoku constraints

    Parameters:
    board: 2d list of integers representing all the tiles in each row of the puzzle
    row: the row of the tile being considered
    col: the column of the tile being considered
    num: the given number

    Returns: Boolean
    (True if number can be placed in tile and satisfy constraints, False otherwise)
    '''
    for i in range(0, 9):
        if board[row][i] == num:
            # Same number in current row -> illegal
            return False
    for i in range(0, 9):
        if board[i][col] == num:
            # Same number in current column -> illegal
            return False
    square_row = (row//3)*3
    square_col = (col//3)*3
    for i in range(0, 3):
        for j in range(0, 3):
            if board[square_row+i][square_col+j] == num:
                # Same number in current 3x3 box -> illegal
                return False
    return True


def solve_sudoku(puzzle):
    ''' 
    Description:
    A function to solve the sudoku puzzle

    Parameters:
    puzzle: 2d list of integers representing all the tiles in each row of the puzzle

    Returns:
    None
    '''
    # Iterate through all the board tiles
    for row in range(9):
        for col in range(9):
            if puzzle[row][col] == 0:
                # Empty tile found! Trying to find a legal move...
                for num in range(1, 10):
                    if is_legal(puzzle, row, col, num):
                        # Legal move found! Update tile and recurse
                        puzzle[row][col] = num
                        solve_sudoku(puzzle)
                        # Backtracking...
                        puzzle[row][col] = 0
                return
    # Print the solved sudoku puzzle
    print('\n Solution:')
    print_board(puzzle)


# Print unsolved sudoku board
print_board(board)
# Solve sudoku and print board
solve_sudoku(board)
