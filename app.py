'''
Flask based REST API for roadagram to HD map matching service.
'''
import json
from flask import Flask, request, abort, Response, jsonify

import rdg.matcher.matcher as matcher

app = Flask(__name__)

@app.route('/')
def index():
  return jsonify({'error': 'Use /match endpoint'}), 404

@app.route('/match', methods=['POST'])
def match_json():
  content = request.json
  if content is None:
    return jsonify({'error': 'Bad content'}), 201

  roadagram = content['roadagram']
  print('Roadagram:\n', roadagram)

  hdmap = content['hdmap']
  print('HDMap:\n', hdmap)

  if roadagram is None or hdmap is None:
    return jsonify({'error': 'Missing roadagram or hdmap attribute'}), 201

  out = matcher.match(roadagram, hdmap)

  return jsonify(out), 200
