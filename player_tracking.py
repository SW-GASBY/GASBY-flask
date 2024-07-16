import json
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from json_convert import json_convert

class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1

    def predict(self):
        pred = self.kalman.predict()
        return pred[0], pred[1]

    def correct(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measurement)

class Player:
    def __init__(self, player_id, initial_position, bbox, position_name, uniform_color):
        self.player_id = player_id
        self.kalman_filter = KalmanFilter()
        self.kalman_filter.correct(initial_position[0], initial_position[1])
        self.position = initial_position
        self.position_name = position_name
        self.bbox = bbox
        self.uniform_color = uniform_color
        self.missed_frames = 0

def get_player_positions(detections):
    positions = []
    bboxes = []
    position_names = []
    uniform_colors = []
    for detection in detections:
        if detection['name'] == 'player' and detection['confidence'] > 0.5:
            box = detection['box']
            x_center = (box['x1'] + box['x2']) / 2
            y_center = (box['y1'] + box['y2']) / 2
            positions.append((x_center, y_center))
            bboxes.append((box['x1'], box['y1'], box['x2'], box['y2']))
            position_names.append(detection['position_name'])
            uniform_colors.append(detection['uniform_color'])
    return positions, bboxes, position_names, uniform_colors

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def match_players(predicted_positions, curr_bboxes):
    cost_matrix = np.zeros((len(predicted_positions), len(curr_bboxes)))
    for i, pred in enumerate(predicted_positions):
        for j, curr in enumerate(curr_bboxes):
            cost_matrix[i, j] = -compute_iou(pred, curr)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind, cost_matrix

MAX_MISSED_FRAMES = 5
def track_players(players, player_id_counter, curr_positions, curr_bboxes, position_names, uniform_colors):
    if not players:
        for pos, bbox, position_name, uniform_color in zip(curr_positions, curr_bboxes, position_names, uniform_colors):
            players.append(Player(player_id_counter, pos, bbox, position_name, uniform_color))
            player_id_counter += 1
    else:
        predicted_positions = [player.kalman_filter.predict() for player in players]
        row_ind, col_ind, cost_matrix = match_players([player.bbox for player in players], curr_bboxes)
        assigned = set()
        for r, c in zip(row_ind, col_ind):
            if -cost_matrix[r, c] > 0.3:
                players[r].kalman_filter.correct(curr_positions[c][0], curr_positions[c][1])
                players[r].position = curr_positions[c]
                players[r].bbox = curr_bboxes[c]
                players[r].missed_frames = 0
                players[r].position_name = position_names[c]
                players[r].uniform_color = uniform_colors[c]
                assigned.add(c)
        for i, (pos, bbox, position_name, uniform_color) in enumerate(zip(curr_positions, curr_bboxes, position_names, uniform_colors)):
            if i not in assigned:
                players.append(Player(player_id_counter, pos, bbox, position_name, uniform_color))
                player_id_counter += 1
        for player in players:
            if player.position not in curr_positions:
                player.missed_frames += 1
        players[:] = [player for player in players if player.missed_frames <= MAX_MISSED_FRAMES]
    return player_id_counter

def player_tracking(source, teamA, teamB):
    with open(source + '/data.json') as f:
        frames = json.load(f)

    tracked_results = []
    players = []
    player_id_counter = 0
    max_missed_frames = 5

    for frame_index, frame_data in enumerate(frames):
        curr_positions, curr_bboxes, position_names, uniform_colors = get_player_positions(frame_data)
        player_id_counter = track_players(players, player_id_counter, curr_positions, curr_bboxes, position_names, uniform_colors)
        frame_results = []
        for player in players:
            frame_results.append({
                'player_id': player.player_id,
                'position_name': player.position_name,
                'position': player.position,
                'box': player.bbox,
                'uniform_color': player.uniform_color
            })
        tracked_results.append(frame_results)

    with open(source + '/tracked_results.json', 'w') as f:
        json.dump(tracked_results, f, indent=4)
    json_convert(source, teamA, teamB)