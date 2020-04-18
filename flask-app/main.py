import os, sys
import time
import datetime
import json
import uuid

from flask import Flask, redirect, url_for, render_template, Markup
from flask import request, jsonify

app = Flask(__name__)
app.config["DEBUG"] = True

from percolation_grid import GridMode0

GAME_STATE = {
    'test': {'size': (50, 50), 'p':0.35},
}

@app.route('/newgame', methods=['GET'])
def get_random_newgame():
    grid = GridMode0()
    grid.make_randomized_alive()

    edgeList = list()
    for i in range(grid.states.shape[0]):
        for j in range(grid.states.shape[1]):
            edgeList.append((i,j,grid.states[i,j]))
    return jsonify({'states':edgeList, 'size': list(grid.states.shape)})

@app.route('/game/<string:game_id>', methods=['GET'])
def get_game_state(game_id):
    return GAME_STATE[game_id]

@app.route('/')
def index_page():
    return render_template('index.html')

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

@app.errorhandler(Exception)
def exception_handler(error):
    import traceback
    trb = traceback.print_tb(error.__traceback__)
    return "!!!!"  + repr(error) + "<br/>" + repr(trb)

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=4000)