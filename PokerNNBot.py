from pypokerengine.players import BasePokerPlayer
from effective_hand_strength import HandStrengthEval
from decimal import Decimal
import random as rand
import numpy as np
import pprint

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / (np.sum(np.exp(x), axis=0) + 1e-9)

def backward_softmax(x):
    return x*(1-x)

def random_choice(dist):
    rnd = rand.random()
    i = 0
    sum = 0
    while rnd < sum and i < len(dist):
        sum += dist[i]
        i += 1
    return i

class Neuron:
    """
    Simple perceptron
    """
    def __init__(self):
        self.weights = np.random.rand(3, 4)
        self.bias = np.random.rand(1, 3)
        self.history = []

    def forward(self, pot, bb_pos, amount, hand_strength):
        inp = np.array([pot/2000, bb_pos, amount/2000, hand_strength])
        out = np.matmul(self.weights,inp) + self.bias
        dist = softmax(out[0])
        print(dist)
        choice = random_choice(dist)
        self.history.append((choice, inp))
        return ["raise", "fold", "call"][choice]

    def backward(self, reward, lr=0.1):
        error_b = np.zeros(shape=(1, 3))
        error_w = np.zeros(shape=(3, 4))
        for choice, inp in self.history:
            out = np.matmul(self.weights, inp) + self.bias
            dist = softmax(out[0])
            arr = [0, 0, 0]
            arr[choice] = 1
            arr = np.array(arr)
            if reward < 0:
                ones = np.array([1,1,1])
                error = dist - (ones-arr)/2
            else:
                error = arr - dist
            res = np.multiply(backward_softmax(out), error)

            error_b -= res
            error_w -= np.outer(res, np.array(inp))
        if len(self.history) == 0:
            return
        self.weights += error_w*lr/len(self.history)
        self.bias += error_b*lr/len(self.history)
        self.history = []
        return


class NNPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_pp = pprint.PrettyPrinter(indent=2)
        bb_pos = round_state["big_blind_pos"]
        pot = round_state["pot"]["main"]["amount"]
        hand_strength = HandStrengthEval()
        hand = hand_strength.hand_strength(list(map(str, hole_card)), [])
        return self.nn.forward(pot, bb_pos, self.amount, hand)

    def receive_game_start_message(self, game_info):
        self.nn = Neuron()
        self.amount = 1000

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count
        self.amount_if_loose = self.amount

    def receive_street_start_message(self, street, round_state):
        print ("Street")

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        #print(winners, round_state)
        if winners[0]["uuid"] == self.uuid:
            print ("YAY! I won!")
            self.amount += round_state["pot"]["main"]["amount"]
            self.nn.backward(1)
        else:
            self.amount = self.amount_if_loose
            print ("Dobby must punish himself...")
            self.nn.backward(-11)


def setup_ai():
  return NNPlayer()
