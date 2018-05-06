'''
Flask based REST API for roadagram to HD map matching service.
'''
from flask import Flask, request, abort, Response, jsonify

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