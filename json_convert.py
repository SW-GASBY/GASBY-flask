# Refactoring into one object using categorized files
import json

def json_convert(source):
    # Load the tracked results data from the file
    file_path = source + '/tracked_results.json'
    with open(file_path, 'r') as file:
        tracked_data = json.load(file)

    # Initialize a dictionary to store positions by player_id
    player_positions = {}

    # Iterate through each frame in the data
    for frame_number, frame_data in enumerate(tracked_data):
        for player in frame_data:
            player_id = player['player_id']
            position = player['position']
            if player_id not in player_positions:
                player_positions[player_id] = {'player_id': player_id, 'position': []}
            player_positions[player_id]['position'].append({'frame': frame_number, 'position': position})

    # Filter out players with positions less than or equal to 20 frames
    filtered_player_positions = [player for player in player_positions.values() if len(player['position']) > 20]

    # Save the filtered result to a new JSON file
    output_file_path = source + '/player_positions_filtered.json'
    with open(output_file_path, 'w') as output_file:
        json.dump(filtered_player_positions, output_file, indent=4)

    print(f'Filtered data saved to {output_file_path}')