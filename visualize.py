import tkinter as tk
import chess

def reset_board():
    # Reset the chessboard to the initial position
    board.set_fen(chess.STARTING_FEN)
    board_label['text'] = str(board)

def make_move():
    # Get the move from the textbox and make the move on the chessboard
    move = textbox.get()
    try:
        board.push_san(move)
        board_label['text'] = str(board)
    except ValueError:
        # Handle invalid moves
        print("Invalid move!")

# Create the main window
window = tk.Tk()
window.title("Chess Game")

# Create the chessboard
board = chess.Board()

board_label = tk.Label(window, text=str(board), font=('Courier', 10))
board_label.pack()

# Create the textbox for the next move
textbox = tk.Entry(window)
textbox.pack()

# Create the button to make the move
move_button = tk.Button(window, text="Make move", command=make_move)
move_button.pack()

# Create the button to reset the board
reset_button = tk.Button(window, text="Reset board", command=reset_board)
reset_button.pack()

window.mainloop()