# Classify objects and group them together with people who are considered similar.
import json
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from json_convert import json_convert

class KalmanFilter:
    def __init__(self):
        # Initialize the Kalman Filter with the required matrices
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1

    def predict(self):
        # Predict the next position based on the current state
        pred = self.kalman.predict()
        return pred[0], pred[1]

    def correct(self, x, y):
        # Correct the Kalman Filter with the new measurements
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measurement)

class Player:
    def __init__(self, player_id, initial_position, bbox, position_name):
        # Initialize a player with a unique ID and initial position
        self.player_id = player_id
        self.kalman_filter = KalmanFilter()
        self.kalman_filter.correct(initial_position[0], initial_position[1])
        self.position = initial_position
        self.position_name = position_name
        self.bbox = bbox
        self.missed_frames = 0

def get_player_positions(detections):
    # Extract player positions and bounding boxes from the detection results
    positions = []
    bboxes = []
    position_names = []
    for detection in detections:
        if detection['name'] == 'player' and detection['confidence'] > 0.5:  # Confidence threshold
            box = detection['box']
            x_center = (box['x1'] + box['x2']) / 2
            y_center = (box['y1'] + box['y2']) / 2
            positions.append((x_center, y_center))
            bboxes.append((box['x1'], box['y1'], box['x2'], box['y2']))
            position_names.append(detection['position_name'])
    return positions, bboxes, position_names

def compute_iou(boxA, boxB):
    # Compute the Intersection over Union (IoU) of two bounding boxes
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
    # Match players between predicted and current bounding boxes using IoU
    cost_matrix = np.zeros((len(predicted_positions), len(curr_bboxes)))

    for i, pred in enumerate(predicted_positions):
        for j, curr in enumerate(curr_bboxes):
            cost_matrix[i, j] = -compute_iou(pred, curr)  # Use negative IoU as cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind, cost_matrix

players = []
player_id_counter = 0
max_missed_frames = 5

def track_players(prev_positions, prev_bboxes, curr_positions, curr_bboxes, position_names):
    global player_id_counter
    if not players:
        # If no players are being tracked, initialize them
        for pos, bbox, position_name in zip(curr_positions, curr_bboxes, position_names):
            players.append(Player(player_id_counter, pos, bbox, position_name))
            player_id_counter += 1
    else:
        # Predict player positions and match them with the current positions
        predicted_positions = [player.kalman_filter.predict() for player in players]
        row_ind, col_ind, cost_matrix = match_players([player.bbox for player in players], curr_bboxes)

        assigned = set()
        for r, c, position_name in zip(row_ind, col_ind, position_names):
            if -cost_matrix[r, c] > 0.3:  # IoU threshold to consider a match valid
                players[r].kalman_filter.correct(curr_positions[c][0], curr_positions[c][1])
                players[r].position = curr_positions[c]
                players[r].bbox = curr_bboxes[c]
                players[r].missed_frames = 0
                players[r].position_name = position_name
                assigned.add(c)

        # Add new players for unmatched positions
        for i, (pos, bbox, position_name) in enumerate(zip(curr_positions, curr_bboxes, position_names)):
            if i not in assigned:
                players.append(Player(player_id_counter, pos, bbox, position_name))
                player_id_counter += 1

        # Update missed frames and remove players that are no longer in the current frame for a certain period
        for player in players:
            if player.position not in curr_positions:
                player.missed_frames += 1

        players[:] = [player for player in players if player.missed_frames <= max_missed_frames]

# if __name__ == "__main__":
def player_tracking(source):
    # Read data from data.json file
    with open(source + '/data.json') as f:
        frames = json.load(f)

    tracked_results = []

    # Process each frame and track player positions
    for frame_index, frame_data in enumerate(frames):
        curr_positions, curr_bboxes, position_names = get_player_positions(frame_data)
        prev_positions = [player.position for player in players]
        prev_bboxes = [player.bbox for player in players]
        
        track_players(prev_positions, prev_bboxes, curr_positions, curr_bboxes, position_names)

        # Store the current positions of each player
        frame_results = []
        for player in players:
            frame_results.append({
                'player_id': player.player_id,
                'position_name' : player.position_name,
                'position': player.position,
                'box': player.bbox
            })
        tracked_results.append(frame_results)

    # Save the tracked results to a JSON file
    with open(source + '/tracked_results.json', 'w') as f:
        json.dump(tracked_results, f, indent=4)
    json_convert(source)