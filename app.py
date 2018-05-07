'''
Flask based REST API for roadagram to HD map matching service.
'''
import json
from flask import Flask, request, abort, Response, jsonify

import rdg.matcher.matcher as matcher

app = Flask(__name__)

@app.route('/')
def index():
  return 'Under development'

@app.route('/match', methods=['POST'])
def match():
  content = request.json
  print('Content:', content)
  if content is None:
    return jsonify({'error': 'Bad content'}), 201
  return jsonify({'id': content['username']}), 200

@app.route('/match_j', methods=['POST'])
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

  #return jsonify({'error': '0'}), 200
  return jsonify(out), 200

@app.route('/match_f', methods=['POST'])
def match_multipart():
  # loading as form attributes
  # roadagram = request.form.get('roadagram)
  # hdmap = request.form.get('hdmap')

  roadagram = request.files['roadagram']
  hdmap = request.files['hdmap']
  print(roadagram)
  if roadagram is None or hdmap is None:
    return jsonify({'error': 'Bad content'}), 201
  lines = ''.join([line for line in roadagram])
  roadagram = json.loads(lines)
  print(roadagram)
  hdmap = json.loads(hdmap)
  
  print('Roadagram:\n', roadagram)

  return jsonify({'error': 0}), 200