'''
Nicholas Dahdah, Charles David, Thomas Jarvis
Final Project
R. Vincent
May 6th, 2019

Learning Boxes and Dots through reinforcement learning.
Learns to play the game over time.
Reinforcement Learning code adapted from R. Vincent's Tic-Tac-Toe
reinforcement learning example.
'''

import random

# These define the reward function.
R_CONTINUE = -5.0
R_WIN = 0.0
R_LOSS = -100.0
R_TIE = -50.0
R_BOX = -1.0

class generic_player(object):
    '''Generic player. Base object of all other player classes.'''
    def __init__(self, game, player_num):
        self.game = game
        self.player_num = player_num
        self.state = None
    
    def learn(self, r, s1, s0):
        '''Generic players never learn.'''
        pass

    def policy(self, s0, boxes):
        '''Select a move at random.'''
        actions = self.game.legal_moves(s0)
        return random.choice(actions)

    def message(self, text):
        '''Prints a message, if appropriate.'''
        pass

    def game_over(self, r):
        '''Game over.'''
        pass

    def winloss(self):
        '''Return the ratio of wins out of all games played.'''
        return 0.0

    def visited(self):
        '''Return the number of states visited.'''
        return 0

class tdzero_player(generic_player):
    '''Learns to play tic-tac-toe using basic temporal difference
    learning, known as TD(0).'''
    def __init__(self, game, player_num):
        super().__init__(game, player_num)
        self.V = {}
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.05
        self.games = 0
        self.won = 0

    def getV(self, s):
        '''Return the estimated value of the given state.'''
        s_str = ""
        for i in s:
            s_str += str(i)
        return self.V.get(s_str, R_WIN)
    
    def learn(self, r, s1, s0):
        '''TD(0) update for value function.'''
        s0_str = ""
        s1_str = ""
        for i in s0:
            s0_str += str(i)
        for i in s1:
            s1_str += str(i)
        v0 = self.getV(s0_str)
        v1 = self.getV(s1_str)
        self.V[s0_str] = v0 + self.alpha * (r + self.gamma * v1 - v0)

    def policy(self, s0, boxes):
        '''Returns a move chosen by an epsilon-greedy policy.'''
        actions = self.game.legal_moves(s0)
        if len(actions) == 0:
            print("no legal moves")
            return -1

        if random.random() < self.epsilon:
            best_action = random.choice(actions)
        else:
            # select a greedy move, breaking ties randomly.
            best_value = -float('inf')
            best_action = -1
            n_ties = 1
            for action in actions:
                s1 = self.game.make_move(s0, action, self.player_num)
                value = self.getV(s1)
                if value > best_value:
                    best_value = value
                    best_action = action
                    n_ties = 1
                elif value == best_value:
                    # break the tie to encourage exploration.
                    n_ties += 1
                    if random.random() < 1 / n_ties:
                        best_value = value
                        best_action = action
        return best_action

    def visited(self):
        '''Return the number of states visited.'''
        return len(self.V)

    def winloss(self):
        '''Return the winning ratio.'''
        return 0 if self.games == 0 else self.won / self.games

    def game_over(self, r):
        '''Keep track of whether we won, and
        print a message if appropriate.'''
        if r == R_WIN:
            self.won += 1
        self.games += 1
    
class human_player(generic_player):
    '''This class represents the interactive human player.'''
    
    def policy(self, s0, boxes):
        '''Select a move interactively.'''
        self.message(boxes)
        while True:
            m = self.game.legal_moves(s0)
            try:
                l = input("Pick a line: ")
                n = int(l) - 1
            except ValueError:
                print("Could not convert data to integer")
                continue

            if n not in m:
                print("illegal move")
            else:
                break
        return n

    def message(self, boxes):
        '''Print a message so the human can see it.'''
        print(boxes)

    def game_over(self, r):
        '''Keep track of whether we won, and
        print a message if appropriate.'''
        if r == R_WIN:
            print("Player " + str(self.player_num) + ": You Win.")
        elif r == R_LOSS:
            print("Player " + str(self.player_num) + ": You Lose.")
        else:
            print("Player " + str(self.player_num) + ": You Tie.")
    
class BoxesAndDots_rl(object):
    '''Boxes and Dots class initializes and handles the game, game learning,
    and board.'''
    def __init__(self, dimX, dimY):
        '''Initializes the game with a set of dimensions for the game board'''
        def buildBoxes(width,height):
            '''Creates a list of all the existing boxes where each box is
            represented as a list of the vertex nodes of that box'''
            counter = 0
            boxList = [list() for i in range((width-1)*(height-1))]
            for x in range(width*height):
                if (x+1)%width==0:
                    continue
                elif x>=(height*width-width):
                    continue
                else:
                    boxList[counter].append(x)
                    boxList[counter].append(x+1)
                    boxList[counter].append(x+width)
                    boxList[counter].append(x+width+1)
                    counter+=1
            return boxList
        def drawBoard(width,height):
            '''Creates a list of strings representing the board, used by the
            repr function.'''
            connectionsList =[]
            for x in range((2*width)*(height)-1):
                if (x+1)%(2*width)==0:
                    for x in range(width-1):
                        connectionsList.append(" ")
                        connectionsList.append("   ")
                    connectionsList.append(" ")
                elif x%2==0:
                    connectionsList.append("0")
                else:
                    connectionsList.append("   ")
            return connectionsList
        
        self._width = dimX
        self._height = dimY
        self._vertex = dimX*dimY
        self._edge = 0
        self._turn = 0
        self._boxcreated = False
        self._adj = [list() for i in range(self._vertex)]
        self._state = [0 for i in range ((dimX-1)*dimY +(dimY-1)*dimX)]
        self._boxes = buildBoxes(self._width, self._height)
        self._board = drawBoard(self._width, self._height)
        self._points = [0 for i in range(len(self._boxes))]
            
    def addEdge(self,s0, a):
        '''Adds edge between two vertices and updates board'''
        #converts action to initial and final nodes
        if a%(self._width*2-1) < self._width-1:
            initial = a-(a//(self._width*2-1))*(self._width-1)
            final = initial + 1
        else:
            initial = a-(a//(self._width*2-1)+1)*(self._width-1)
            final = initial + self._width
        # if legal move, adds line to board
        if a in boxes.legal_moves(s0):
            self._adj[initial].append(final)
            self._adj[final].append(initial)
            self._edge += 1
            # updates board
            if final - initial == 1:
                self._board[initial*2+1+(2*self._width-2)*(initial//self._width)] = "---"
            else:
                self._board[initial*2+(2*self._width-2)*(initial//self._width)+1*(2*self._width-1)] = "|"
            boxes.boxCheck(initial,final)
            # updates turn
            self._turn = 1 - self._turn
        else:
            print("non-legal move")


    def boxCheck(self, initial, final):
        '''Checks if a box has been completed, and updates the board if
        there has been.'''
        boxesToCheck = []
        box_created = False
        for box in self._boxes:
            if (initial in box) and (final in box):
                boxesToCheck.append(box)
        for box in boxesToCheck:
            # checks to see if box is surrounded by edges
            if box[1] in self._adj[box[0]] and box[2] in self._adj[box[0]] and box[1] in self._adj[box[3]] and box[2] in self._adj[box[3]]:
                self._points[self._boxes.index(box)] = self._turn*2-1 #updates points list
                self._board[box[0]*2+(2*self._width-2)*(box[0]//self._width)+2*self._width] = " " + str(self._turn) + " " #updates board
                box_created = True
        if box_created:
            self._boxcreated = self._turn
            self._turn = 1 - self._turn # resets turn
        else:
            self._boxcreated = -1

    def winner(self):
        '''Returns the winner of the game by adding up points list.'''
        box_sum = 0
        for i in self._points:
            box_sum += i
        if box_sum > 0:
            return 1
        elif box_sum == 0:
            return -1
        else:
            return 0

    def legal_moves(self, state):
        '''Returns a list of all possible, legal actions'''
        legal_moves = []
        for i in range(len(state)):
            if state[i] == 0:
                legal_moves.append(i)
        return legal_moves

    def make_move(self, state, action, player):
        '''Updates the state with an action'''
        new_state = state.copy()
        new_state[action] = 1
        return new_state
        
    def done(self):
        '''Determines whether or not the game is completed'''
        if boxes._edge < (boxes._width-1)*(boxes._height) + (boxes._height-1)*(boxes._width):
            return False
        else:
            return True

    def reward(self, p1):
        '''Returns a reward for a completed action'''
        if p1 == 0:
            p2 = 1
        else:
            p2 = 0
        if self.done():
            if self.winner() == p1:
                return R_WIN
            elif self.winner() == p2:
                return R_LOSS
            else:
                return R_TIE
        if self._boxcreated == p1:
            return R_BOX
        return R_CONTINUE

    def start(self):
        '''Initialize the state of the game.'''
        return self._state

    def __repr__(self):
        '''Prints board'''
        board_string = ""
        counter = 0
        for i in self._board:
            counter += 1
            if counter%(2*self._width-1) == 1:
                board_string += "\n"
            board_string += i
        return board_string

    @staticmethod
    def game(players):
        '''Plays one game.'''
        boxes.__init__(boxes._width, boxes._height) # reinitializes game
        s0 = boxes.start()
        n_player = 0
        finished = False
        while not finished:
            player = players[n_player]
            a = player.policy(s0, boxes) # finds action
            s1 = boxes.make_move(s0, a, player.player_num) # takes action
            boxes.addEdge(s0,a) # updates board and state
            boxes._state = s1
            # both players learn from all moves.
            for player in players:
                r = boxes.reward(player.player_num) # get the reward.
                player.learn(r, s1, s0)
                if r != R_CONTINUE and r != R_BOX:
                    finished = True
                    player.game_over(r)
            s0 = s1
            n_player = boxes._turn
        

    @staticmethod
    def play(players, n_games,  player_test = None):
        '''Plays multiple games'''
        i = 0
        if not player_test == None:
            boxes.play_sample([players[0],player_test], 1000, n_games, i)
        while i != n_games:
            i += 1
            if i%10000 == 0 and not player_test == None: #samples every 10000 games
                boxes.play_sample([players[0],player_test], 1000, n_games, i)
            boxes.game([players[0],players[1]])

    @staticmethod
    def play_sample(players, n_games, n_games_total, n_games_done):
        '''Plays AI against random opponent and prints Win/Loss ratio
        as well as % training complete'''
        players[0].epsilon = 0.0
        players[0].games = 0
        players[0].won = 0
        i = 0
        while i != n_games:
            i += 1
            boxes.game(players)
        print("AI training ... "  + str(int((n_games_done/n_games_total)*100//1)) + "% Complete (winrate: " + str((players[0].winloss()//0.001/10)) + "%) ")
        players[0].epsilon = 0.05
        

width,height,human_players = 0,0,0
while width < 2:
    width = int(input("Choose box width: "))
while height < 2:
    height = int(input("Choose box height: "))
while human_players < 1 or human_players > 2:
    human_players = int(input("One or Two players? (1 or 2): "))
boxes = BoxesAndDots_rl(width,height) #initializes board
if human_players == 1:
    order = 0
    # determines who plays first
    while order < 1 or order > 2:
        order = int(input("Would you like to play first or second? (1 or 2): "))
    n_games = int(input("Choose number of games to train over: "))
    p1 = tdzero_player(boxes, 0) # AI player playing first
    p2 = tdzero_player(boxes, 1) # AI player playing second
    p3 = generic_player(boxes,0) # random player for sampling
    boxes.play([p1, p2], n_games, p3) # train actual AI
    p1.epsilon = 0.0 # exploit only.
    p2.epsilon = 0.0
    if order == 2:
        ph = human_player(boxes, 1) # human player playing second
        boxes.play([p1, ph], -1)
    else:
        ph = human_player(boxes, 0) # human player playing first
        boxes.play([ph, p2], -1)
else:
    # human vs human
    p1 = human_player(boxes, 0)
    p2 = human_player(boxes, 1)
    boxes.play([p1, p2], -1)

