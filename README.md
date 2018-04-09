# alpha-zero
AlphaZero implementation based on ["Mastering the game of Go without human knowledge"](https://deepmind.com/documents/119/agz_unformatted_nature.pdf) and ["Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"](https://arxiv.org/abs/1712.01815) by DeepMind.

The algorithm learns to play games like Chess and
Go without any human knowledge. It uses Monte Carlo Tree Search and a Deep Residual Network to evaluate
the board state and play the most promising move.

Games implemented:
1) Tic Tac Toe

## Requirements
 - TensorFlow (Tested on 1.4.0)
 - NumPy
 - Python 3
 
## Usage
**To train the model from scratch.**:
```
python main.py --load_model 0
``` 

**To train the model using the previous best model as a starting point**:
```
python main.py --load_model 1
``` 

**To play a game vs the previous best model**:
```
python main.py --load_model 1 --human_play 1
``` 

**Options**:
* `--num_iterations`: Number of iterations.
* `--num_games`: Number of self play games played during each iteration.
* `--num_mcts_sims`: Number of MCTS simulations per game.
* `--c_puct`: The level of exploration used in MCTS.
* `--l2_val`: The level of L2 weight regularization used during training.
* `--momentum`: Momentum Parameter for the momentum optimizer.
* `--learning_rate`: Learning Rate for the momentum optimizer.
* `--t_policy_val`: Value for policy prediction.
* `--temp_init`: Initial Temperature parameter to control exploration.
* `--temp_final`: Final Temperature parameter to control exploration.
* `--temp_thresh`: Threshold where temperature init changes to final.
* `--epochs`: Number of epochs during training.
* `--batch_size`: Batch size for training.
* `--dirichlet_alpha`: Alpha value for Dirichlet noise.
* `--epsilon`: Value of epsilon for calculating Dirichlet noise.
* `--model_directory`: Name of the directory to store models.
* `--num_eval_games`: Number of self-play games to play for evaluation.
* `--eval_win_rate`: Win rate needed to be the best model.
* `--load_model`: Binary to initialize the network with the best model.
* `--human_play`: Binary to play as a Human vs the AI.
* `--resnet_blocks`: Number of residual blocks in the resnet.
* `--record_loss`: Binary to record policy and value loss to a file.
* `--loss_file`: Name of the file to record loss.
* `--game`: Number of the game. 0: Tic Tac Toe, 1: Othello.

## License
    MIT License

    Copyright (c) 2018 Blanyal D'Souza

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
