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
window.geometry("300x300")
# window.

# Create the chessboard
board = chess.Board()

board_label = tk.Label(window, text=str(board), font=('Courier', 10))
board_label.pack()

# Create the textbox for the next move
textbox = tk.Entry(window)
textbox.pack(padx=5, pady=5)

# Create the button to make the move
move_button = tk.Button(window, text="Make move", command=make_move, background='green')
# move_button.pack(pady=10)
move_button.pack(pady=10, side='left')

# Create tje button for the model to make the move
bot_move_button = tk.Button(window, text="Model move", command=make_move, background='blue')
bot_move_button.pack(pady=10)

# Create the button to reset the board
reset_button = tk.Button(window, text="Reset board", command=reset_board, background='red')
reset_button.pack()

window.mainloop()