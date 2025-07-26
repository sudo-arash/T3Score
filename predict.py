import numpy as np
from T3Score import load_model

def board_to_input_tensor(board):
    """
    Convert a 3x3 board with values {1, -1, 0} into a 3x3x3 one-hot tensor
    Channels: Player 1, Player 2, Empty
    """
    assert board.shape == (3,3), "Board must be 3x3"
    assert np.all(np.isin(board, [-1, 0, 1])), "Board values must be -1, 0, or 1"

    tensor = np.zeros((3,3,3), dtype=np.float32)
    for r in range(3):
        for c in range(3):
            val = board[r, c]
            if val == 1:
                tensor[r, c, 0] = 1  # Player 1 channel
            elif val == -1:
                tensor[r, c, 1] = 1  # Player 2 channel
            else:
                tensor[r, c, 2] = 1  # Empty channel
    return tensor

def predict_board_score(board, net=None):
    """
    Predict the board score using the loaded network.
    Args:
        board: 3x3 numpy array with values {1, -1, 0}
        net: the keras model. If None, it loads it automatically.
    Returns:
        score: float, scaled from 0 to 10
    """
    if net is None:
        net = load_model()

    input_tensor = board_to_input_tensor(board)
    # Model expects input shape (batch, height, width, channels)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    pred_scaled = net.predict(input_tensor)[0][0]  # extract scalar from batch output

    score = pred_scaled * 10
    return score
