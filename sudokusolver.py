def solve(board):
    solvePartialSudoku(0, 0, board)
	
    return board


def solvePartialSudoku(row, col, board):
	currentRow = row 
	currentCol = col 
	if currentCol == len(board[currentRow]):
		currentRow+=1 
		currentCol=0 
		if currentRow == len(board):
			return True 
	if board[currentRow][currentCol]==0:
		return tryDigitsAtPosition(currentRow, currentCol, board)
	return solvePartialSudoku(currentRow, currentCol+1,board)
def tryDigitsAtPosition(row, col, board):
	for digit in range(1, 10):
		if isValidAtPosition(digit, row, col, board):
			board[row][col]=digit
			#we check if its the last needed digit
			if solvePartialSudoku(row, col+1,board):
				return True 
	# if we made a mistake we reset to backtrack
	board[row][col]=0
	return False
def isValidAtPosition(value, row, col, board):
	#check if the value is in the row
	rowIsValid = value not in board[row]
	#check if the value is in that column
	columnIsValid = value not in map(lambda r:r[col],board)
	if not rowIsValid or not columnIsValid:
		return False 
	#chrck the subgrid now
	subgridRowStart = (row//3)*3 
	subgridColStart = (col//3)*3 
	for rowIdx in range(3):
		for colIdx in range(3):
			# check columbs and rows 9 times 
			rowToCheck = subgridRowStart + rowIdx
			colToCheck = subgridColStart+ colIdx 
			existingValue = board[rowToCheck][colToCheck]
			# if our value exists in the subgrid its invalid
			if existingValue == value:
				return False 
	return True 
