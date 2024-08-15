import pygame
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import sys
import pickle

# Initialize the Pygame module
pygame.init()

# Define display dimensions
DISPLAY_WIDTH, DISPLAY_HEIGHT = 680, 700
window = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption('Space Adventure AI')

# Define color constants
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# Set up game timing
game_clock = pygame.time.Clock()

# Player (spaceship) properties
player = pygame.Rect(300, 300, 30, 30)
player_speed = 0
gravity_force = 0.5

# Obstacles properties
obstacle_width = 40
obstacle_gap = 120
obstacle_movement_speed = -3

# Load game assets
def load_assets():
    try:
        ship_img = pygame.image.load(r'sources\images\alien-ufo-scene.png')
        ship_img = pygame.transform.scale(ship_img, (45, 40))  # Adjust size

        top_obs = pygame.image.load(r'sources\images\pipe2.png')
        bottom_obs = pygame.image.load(r'sources\images\pipe.png')

        bg_img = pygame.image.load(r'sources\images\backgroung.jpg')
        bg_img = pygame.transform.scale(bg_img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        start_img = pygame.image.load(r'sources\images\ieyp_0tjn_210901.jpg')
        start_img = pygame.transform.scale(start_img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        pygame.mixer.music.load(r'sources\audios\Spaceship Engine Start Up Sound Effect - FREE DOWNLOAD.mp3')
        jump_fx = pygame.mixer.Sound(r'sources\audios\633247__aesterial-arts__arcade-jump-2.wav')
        crash_fx = pygame.mixer.Sound(r'sources\audios\37732__longhairman__1khz_1ds_peep.wav')

        return ship_img, top_obs, bottom_obs, bg_img, start_img, jump_fx, crash_fx
    except pygame.error as e:
        print(f"Failed to load assets: {e}")
        pygame.quit()
        sys.exit()

spaceship_img, obstacle_img, obstacle_img_flip, background_img, start_screen_img, jump_sound, collision_sound = load_assets()

# Define the AI model for game control
def build_ai_model():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(8,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

ai_model = build_ai_model()

# Function to calculate game state
def evaluate_state(player, obstacles, player_speed):
    player_y = player.y / DISPLAY_HEIGHT
    player_speed /= 10
    obs_x = obstacles[0][0].x / DISPLAY_WIDTH
    
    top_obs_bottom = obstacles[0][1].top
    bottom_obs_top = obstacles[0][0].bottom
    
    obs_gap_top = top_obs_bottom / DISPLAY_HEIGHT
    obs_gap_bottom = bottom_obs_top / DISPLAY_HEIGHT
    obs_gap_size = (top_obs_bottom - bottom_obs_top) / DISPLAY_HEIGHT
    
    player_height = player.height / DISPLAY_HEIGHT
    player_width = player.width / DISPLAY_WIDTH

    if player.y > bottom_obs_top:
        distance_to_gap = player.y - bottom_obs_top
    else:
        distance_to_gap = top_obs_bottom - player.y

    gap_size = top_obs_bottom - bottom_obs_top
    normalized_gap_distance = distance_to_gap / gap_size
    
    return np.array([
        player_y,
        player_speed,
        obs_x,
        obs_gap_top,
        obs_gap_bottom,
        obs_gap_size,
        player_height,
        normalized_gap_distance
    ])

# Function to determine reward based on player's position
def calculate_reward(player, obstacles):
    top_obs = obstacles[0][0]
    bottom_obs = obstacles[0][1]

    if player.colliderect(top_obs) or player.colliderect(bottom_obs) or player.y > DISPLAY_HEIGHT or player.y < 0:
        return -1

    gap_top = top_obs.bottom
    gap_bottom = bottom_obs.top

    player_center = player.y + player.height / 2
    distance_from_gap = player_center - gap_bottom

    gap_size = gap_bottom - gap_top
    normalized_gap_distance = abs(distance_from_gap) / gap_size

    if player.y > gap_top and (player.y + player.height) < gap_bottom:
        reward = 1 - normalized_gap_distance
    else:
        reward = 1 - normalized_gap_distance

    return reward

# Generate new obstacle pair
def create_obstacle_pair():
    height = random.randint(100, 400)
    top_obstacle = pygame.Rect(DISPLAY_WIDTH, 0, obstacle_width, height)
    bottom_obstacle = pygame.Rect(DISPLAY_WIDTH, height + obstacle_gap, obstacle_width, DISPLAY_HEIGHT - height - obstacle_gap)
    return top_obstacle, bottom_obstacle

# Render the game elements on the screen
def render_game(player, obstacles, score, mode):
    window.fill(COLOR_WHITE)
    window.blit(background_img, (0, 0))
    window.blit(spaceship_img, (player.x, player.y))
    for obstacle in obstacles:
        top_obs_img = pygame.transform.scale(obstacle_img, (obstacle_width, obstacle[0].height))
        bottom_obs_img = pygame.transform.scale(obstacle_img_flip, (obstacle_width, DISPLAY_HEIGHT - obstacle[0].height - obstacle_gap))
        window.blit(top_obs_img, (obstacle[0].x, obstacle[0].y))
        window.blit(bottom_obs_img, (obstacle[1].x, obstacle[1].y))

    font = pygame.font.Font(None, 36)
    score_display = font.render(f'Score: {score}', True, COLOR_BLACK)
    mode_display = font.render(f'Mode: {mode}', True, COLOR_BLACK)
    window.blit(score_display, (DISPLAY_WIDTH - score_display.get_width() - 10, 10))
    window.blit(mode_display, (10, 10))
    
    pygame.display.flip()

# Reset the game state for a new game
def reset_game():
    global player, player_speed, obstacles, score
    player = pygame.Rect(100, 300, 30, 30)
    player_speed = 0
    obstacles = [create_obstacle_pair()]
    score = 0

# Collect training data by allowing the user to play the game
def gather_training_data():
    global player_speed, score
    training_dataset = []
    reset_game()
    is_running = True
    while is_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                player_speed = -8
                if hasattr(pygame, 'mixer'):
                    jump_sound.play()

        player_speed += gravity_force
        player.y += player_speed

        if obstacles[-1][0].x < -obstacle_width + 300:
            obstacles.append(create_obstacle_pair())
        if obstacles[0][0].x < -obstacle_width:
            obstacles.pop(0)

        for obstacle in obstacles:
            obstacle[0].x += obstacle_movement_speed
            obstacle[1].x += obstacle_movement_speed

        if obstacles[0][0].x < -obstacle_width:
            obstacles.pop(0)
            obstacles.append(create_obstacle_pair())
            score += 1

        reward = calculate_reward(player, obstacles)
        state = evaluate_state(player, obstacles, player_speed)
        training_dataset.append((state, reward))

        if reward == -1:
            if hasattr(pygame, 'mixer'):
                collision_sound.play()
            is_running = False

        render_game(player, obstacles, score, "Player Mode")
        game_clock.tick(30)

    return training_dataset
        
# Train the neural network model using collected data
def train_ai_model(model, training_dataset, epochs=10):
    states, rewards = zip(*training_dataset)
    states = np.array(states)
    rewards = np.array(rewards)
    model.fit(states, rewards, epochs=epochs)

# Load previously saved training data
def load_saved_data():
    try:
        with open('training_data.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return []
    except (pickle.PickleError, EOFError) as e:
        print(f"Error loading data: {e}")
        return []

# Enable the AI to play the game
def ai_play():
    global player_speed, score
    reset_game()
    is_running = True
    while is_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
                pygame.quit()
                sys.exit()

        state = evaluate_state(player, obstacles, player_speed)
        action = ai_model.predict(state.reshape(1, 8), verbose=0)[0][0]

        if action > 0.7:
            player_speed = -8
            if hasattr(pygame, 'mixer'):
                jump_sound.play()

        player_speed += gravity_force
        player.y += player_speed

        for obstacle in obstacles:
            obstacle[0].x += obstacle_movement_speed
            obstacle[1].x += obstacle_movement_speed

        if obstacles[0][0].x < -obstacle_width:
            obstacles.pop(0)
            obstacles.append(create_obstacle_pair())
            score += 1

        if player.colliderect(obstacles[0][0]) or player.colliderect(obstacles[0][1]) or player.y > DISPLAY_HEIGHT:
            if hasattr(pygame, 'mixer'):
                collision_sound.play()
            is_running = False
        
        render_game(player, obstacles, score, "AI Mode")
        game_clock.tick(30)

# Save training data to a file
def save_data(training_dataset, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(training_dataset, file)
    except Exception as e:
        print(f"Error saving data: {e}")

# Start screen setup
def start_screen():
    button_width, button_height = 200, 50
    button_x, button_y = (DISPLAY_WIDTH - button_width) // 2, DISPLAY_HEIGHT // 2
    start_button = pygame.Rect(button_x, button_y, button_width, button_height)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if start_button.collidepoint(mouse_x, mouse_y):
                    return

        window.blit(start_screen_img, (0, 0))
        pygame.draw.rect(window, '#358316', start_button)
        font = pygame.font.Font(None, 36)
        start_text = font.render("Start", True, COLOR_WHITE)
        window.blit(start_text, start_text.get_rect(center=start_button.center))

        pygame.display.flip()
        game_clock.tick(30)

# Main function to run the game
def game_loop():
    all_data = load_saved_data()
    start_screen()

    for i in range(3):
        print(f"Session {i+1}/3: Collecting Data")
        new_data = gather_training_data()
        all_data.extend(new_data)

    if all_data:
        print("Training AI Model")
        train_ai_model(ai_model, all_data, epochs=10)
        print("Starting AI Gameplay")
        ai_play()
        print("Saving Data")
        save_data(all_data, 'training_data.pkl')

if __name__ == "__main__":
    previous_data = load_saved_data()
    if previous_data:
        print("Training AI Model")
        train_ai_model(ai_model, previous_data, epochs=10)
        print("Starting AI Gameplay")
        ai_play()
    game_loop()
