import argparse
import os

class Options():
    def __init__(self, parser):
        self.initialized = False

    def initialize(self, parser):
        
        # Dataset options
        parser.add_argument('--exp', type=str, default='exp/01',         help='name of the exp folder')

        self.initialized = True
        return parser