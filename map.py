# map.py

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import torch # Make sure torch is imported
from collections import deque

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
# from kivy.graphics.texture import Texture # Texture seems unused

# Importing the TD3 Agent components
from td3_agent import TD3, ReplayBuffer

# --- Configuration ---
display_width = 1429
display_height = 660
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'width', str(display_width)) # Use string for config
Config.set('graphics', 'height', str(display_height)) # Use string for config
os.makedirs("./pytorch_models", exist_ok=True) # Ensure model directory exists
os.makedirs("./results", exist_ok=True) # Ensure results directory exists

# --- TD3 Agent Initialization ---
state_dim = 9  # 3 sensors + 2 orientation + 4 boundary distances
action_dim = 1  # 1 action: rotation amount
max_action = 10.0 # Max rotation degrees per step (Increased based on previous suggestion)

policy = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(max_size=1e5) # Smaller buffer size

# Training control parameters
start_timesteps = 1000  # Steps of random actions before using policy
batch_size = 100
expl_noise = 0.01 # Exploration noise scale (Reduced based on previous suggestion)
train_freq = 2 # Train the model every N steps (Changed based on previous suggestion)
train_iterations = 1 # Number of gradient steps per training call
total_timesteps = 0 # Global step counter
scores = [] # List to store scores for plotting (optional)
last_reward = 0 # Tracks reward from previous step
last_distance = 0 # Tracks distance from previous step

# Goal points
goal_points = [(100, 200), (500, 400), (1200, 300)]
current_goal_index = 0
goal_x = goal_points[current_goal_index][0]
goal_y = goal_points[current_goal_index][1]

# Map Initialization Globals
sand = np.zeros((display_height, display_width)) # Initialize sand array
first_update = True

# --- Map Loading Function ---
def init():
    global sand, goal_x, goal_y, first_update
    global display_width, display_height

    try:
        img = PILImage.open("./images/mask.png").convert('L')
        print("Mask image loaded successfully.")
    except FileNotFoundError:
        print("ERROR: mask.png not found in ./images/ directory!")
        img = PILImage.new('L', (display_width, display_height), color=0) # Default black

    # Rotate 90 degrees anti-clockwise
    img = img.rotate(90, expand=True)
    print(f"Image rotated. New size: {img.size}")

    # Resize the *rotated* image
    resized_img = img.resize((display_width, display_height), PILImage.Resampling.LANCZOS)
    print(f"Image resized to: {resized_img.size}")

    # Convert to numpy array and normalize
    sand_temp = np.asarray(resized_img) / 255.0

    # Flip vertically
    sand = np.flipud(sand_temp)
    print(f"Sand array created with shape: {sand.shape}, dtype: {sand.dtype}")

    # Set initial goal
    goal_x = goal_points[current_goal_index][0]
    goal_y = goal_points[current_goal_index][1]

    first_update = False
    print("Initialization complete.")


# --- Car Class ---
class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation_input):
      longueur = display_width
      largeur = display_height

      # --- Determine if car is currently on path ---
      on_path = False
      car_x_int = int(np.clip(self.x, 0, longueur - 1))
      car_y_int = int(np.clip(self.y, 0, largeur - 1))
      try:
          # Check sand value at current center position
          if sand.shape == (largeur, longueur) and sand[car_y_int, car_x_int] < 0.1: # Path threshold
              on_path = True
      except IndexError:
          print(f"IndexError checking sand in move at {car_x_int},{car_y_int}")
      # -------------------------------------------

      # --- Adjust Speed Based on Path ---
      if on_path:
          base_speed = 2.0 # Normal speed on path
      else:
          base_speed = 0.5 # Reduced speed off path (friction) - TUNE THIS VALUE
      # ----------------------------------

      # 1. Set rotation property (casting input)
      current_angle_before_rotation = self.angle
      try:
         self.rotation = float(rotation_input)
      except ValueError as e:
         print(f"ERROR: Invalid rotation value '{rotation_input}': {e}")
         self.rotation = 0.0

      # 2. Update angle
      self.angle = current_angle_before_rotation + self.rotation

      # 3. Calculate velocity using the adjusted base_speed
      current_velocity = Vector(base_speed, 0).rotate(current_angle_before_rotation)

      # Optional: Print velocity info periodically
      # if total_timesteps % 5000 == 0: print(...)

      # 4. Calculate potential new position
      new_pos_vec = Vector(*current_velocity) + self.pos

      # 5. Clamp position using Python min/max
      clamped_x = max(1.0, min(new_pos_vec.x, float(longueur - 1)))
      clamped_y = max(1.0, min(new_pos_vec.y, float(largeur - 1)))

      # 6. Assign final clamped position
      self.pos = Vector(clamped_x, clamped_y)

      # 7. Update sensors based on FINAL position and NEW angle
      self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
      self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
      self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos

      # 8. Calculate sensor signals
      if sand.size == 0 or sand.shape[0] != largeur or sand.shape[1] != longueur:
            print("Warning: Sand array not initialized correctly in sensor calc.")
            self.signal1 = self.signal2 = self.signal3 = 0.5
            return

      for i, sensor_pos in enumerate([self.sensor1, self.sensor2, self.sensor3]):
         sx = int(np.clip(sensor_pos[0], 0, longueur - 1))
         sy = int(np.clip(sensor_pos[1], 0, largeur - 1))
         signal_value = 0.5 # Default if area invalid

         y_min = max(0, sy - 5)
         y_max = min(largeur, sy + 5)
         x_min = max(0, sx - 5)
         x_max = min(longueur, sx + 5)

         if y_max > y_min and x_max > x_min:
               try:
                  signal_value = float(np.mean(sand[y_min:y_max, x_min:x_max]))
               except IndexError:
                  print(f"IndexError accessing sand at {sx},{sy}")
                  signal_value = 0.5
         else:
               try:
                  signal_value = sand[sy, sx] # Fallback single pixel
               except IndexError:
                  print(f"IndexError accessing sand fallback at {sx},{sy}")
                  signal_value = 0.5

         if i == 0: self.signal1 = signal_value
         elif i == 1: self.signal2 = signal_value
         else: self.signal3 = signal_value


# --- Ball Widgets for Sensors (Visualisation) ---
class Ball1(Widget): pass
class Ball2(Widget): pass
class Ball3(Widget): pass


# --- Game Widget (Main Simulation Logic) ---
class Game(Widget):
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Game, self).__init__(**kwargs)
        self.draw_goals()
        # --- Add Stuck Detection Variables ---
        self.stuck_check_window = 100     # How many steps to look back (tune)
        self.stuck_pos_threshold = 10.0   # How much distance counts as 'not stuck' (tune)
        self.stuck_action_threshold = 50 # How many consecutive 'stuck' steps trigger random action (tune)
        self.stuck_counter = 0           # Counts consecutive steps with minimal movement
        self.recent_positions = deque(maxlen=self.stuck_check_window) # Stores recent (x, y) Vectors

    def draw_goals(self):
        with self.canvas.before:
            Color(1, 1, 1)
            for x, y in goal_points:
                Ellipse(pos=(x - 10, y - 10), size=(20, 20))

    def serve_car(self):
        self.car.pos = goal_points[0]
        self.car.velocity = Vector(2, 0)
        self.car.angle = 0

    # Helper to get current state vector
    def _get_current_state(self):
        global goal_x, goal_y, display_width, display_height

        signal1 = self.car.signal1
        signal2 = self.car.signal2
        signal3 = self.car.signal3

        goal_vec = Vector(goal_x, goal_y)
        car_pos_vec = Vector(*self.car.pos)
        car_vel_vec = Vector(*self.car.velocity)
        if car_vel_vec.length() > 0:
            vel_angle = car_vel_vec.angle((1,0))
        else:
            vel_angle = self.car.angle

        goal_dir_vec = goal_vec - car_pos_vec
        orientation = 0
        if goal_dir_vec.length() > 0:
           goal_angle = goal_dir_vec.angle((1,0))
           orientation = (vel_angle - goal_angle)
           orientation = (orientation + 180) % 360 - 180
           orientation /= 180.0

        dist_left = self.car.x / display_width
        dist_right = (display_width - self.car.x) / display_width
        dist_bottom = self.car.y / display_height
        dist_top = (display_height - self.car.y) / display_height

        state = [signal1, signal2, signal3,
                 orientation, -orientation,
                 dist_left, dist_right, dist_bottom, dist_top]

        return np.array(state)


      # Inside Game class in map.py

    # Main update loop, called by Clock
    def update(self, dt):
        global policy, replay_buffer, total_timesteps, current_goal_index, goal_x, goal_y
        global last_reward, last_distance, first_update, scores
        global state_dim, action_dim, max_action, start_timesteps, batch_size, expl_noise, train_freq, train_iterations

        if first_update:
            init()
            if sand.size == 0:
                 print("ERROR: Initialization failed, stopping.")
                 App.get_running_app().stop()
                 return
            self.serve_car()
            last_distance = Vector(*self.car.pos).distance((goal_x, goal_y))
            self.recent_positions.clear() # Clear position history on init
            self.stuck_counter = 0        # Reset stuck counter

        # --- Get Current State ---
        # Get state BEFORE potentially overriding action based on previous steps
        current_state = self._get_current_state()

        # --- Stuck Detection Logic ---
        is_stuck = False
        # Only check if buffer is full and we are past initial random phase
        if total_timesteps > start_timesteps and len(self.recent_positions) == self.stuck_check_window:
            pos_start = self.recent_positions[0] # Oldest position in deque
            pos_end = Vector(*self.car.pos)      # Current position
            distance_moved = pos_start.distance(pos_end)

            if distance_moved < self.stuck_pos_threshold:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0 # Reset if moved enough

            if self.stuck_counter >= self.stuck_action_threshold: # Use >=
                is_stuck = True
                print(f"Stuck detected at step {total_timesteps}! Applying random action.")
                # Reset counter immediately to avoid continuous random actions
                self.stuck_counter = 0
        # --- End Stuck Detection ---

        # --- Select Action (with Stuck Override) ---
        if is_stuck:
            # Override with random action if stuck
            rotation = np.random.uniform(-max_action, max_action)
            # Store the random action taken in the buffer? Or original intended action?
            # Let's store the random action in this case, as it's what was executed.
            action = np.array([rotation])
        elif total_timesteps < start_timesteps:
            # Initial random actions
            rotation = np.random.uniform(-max_action, max_action)
            action = np.array([rotation])
        else:
            # Use TD3 policy action
            action_selected = policy.select_action(current_state)
            noise = np.random.normal(0, expl_noise * max_action, size=action_dim)
            action_noisy = (action_selected + noise).clip(-max_action, max_action)
            rotation = action_noisy[0]
            action = action_selected # Store original policy action in buffer

        # --- Move the car ---
        self.car.move(rotation) # Move using the determined rotation

        # --- Update Position History (AFTER moving) ---
        self.recent_positions.append(Vector(*self.car.pos))
        # ------------------------------------------

        # --- Calculate distance ONCE after moving ---
        distance = Vector(*self.car.pos).distance((goal_x, goal_y))

        # --- Simplified Reward Logic ---
        # ... (Keep the simplified reward logic from previous step here) ...
        new_last_reward = 0.0
        on_path = False
        car_x_int = int(np.clip(self.car.x, 0, display_width - 1))
        car_y_int = int(np.clip(self.car.y, 0, display_height - 1))
        try:
            if sand[car_y_int, car_x_int] < 0.1: on_path = True
        except IndexError: pass

        if on_path: new_last_reward = 0.1
        else: new_last_reward = -10.0

        # --- Boundary Check ---
        margin = 10
        if not (margin <= self.car.x < display_width - margin and margin <= self.car.y < display_height - margin):
            new_last_reward = -50.0
            self.car.angle = (self.car.angle + 180) % 360 # Reverse direction
            self.car.x = max(float(margin), min(self.car.x, float(display_width - margin - 1)))
            self.car.y = max(float(margin), min(self.car.y, float(display_height - margin - 1)))

        # --- Goal Reaching ---
        done_bool = 0.0
        if distance < 25:
             current_goal_index = (current_goal_index + 1) % len(goal_points)
             goal_x = goal_points[current_goal_index][0]
             goal_y = goal_points[current_goal_index][1]
             new_last_reward = 50.0
             print(f"Goal Reached! New target: ({goal_x}, {goal_y})")
             done_bool = 1.0
             # Force reorientation
             new_goal_vec = Vector(goal_x, goal_y)
             car_pos_vec = Vector(*self.car.pos)
             direction_to_new_goal = new_goal_vec - car_pos_vec
             if direction_to_new_goal.length() > 0:
                 self.car.angle = direction_to_new_goal.angle((1,0))
             self.car.velocity = Vector(0,0) # Stop momentarily at goal


        # --- Get Next State ---
        next_state = self._get_current_state()

        # --- Store Transition ---
        # 'action' now correctly holds either the random or policy action
        replay_buffer.add((current_state, next_state, action, new_last_reward, done_bool))

        # --- Train Policy Periodically ---
        if total_timesteps >= start_timesteps and total_timesteps % train_freq == 0:
            if len(replay_buffer.storage) >= batch_size:
                policy.train(replay_buffer, train_iterations, batch_size)

        # --- Update counters and tracking variables ---
        last_reward = new_last_reward
        last_distance = distance
        total_timesteps += 1
        scores.append(new_last_reward)

        # Update sensor visuals
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        # Print status periodically
        if total_timesteps % 500 == 0:
             print(f"Step: {total_timesteps}, Reward: {new_last_reward:.2f}, Distance: {distance:.1f}")


# --- Paint Widget (Optional) ---
class MyPaintWidget(Widget):
    def on_touch_down(self, touch): pass # Disabled
    def on_touch_move(self, touch): pass # Disabled


# --- Kivy Application Class ---
class CarApp(App):

    def build(self):
        parent = Widget()
        self.game = Game()
        # self.game.serve_car() # Call serve_car AFTER init in first update
        Clock.schedule_interval(self.game.update, 1.0 / 60.0)
        parent.add_widget(self.game)

        button_size = (100, 50)
        clearbtn = Button(text='clear', size_hint=(None, None), size=button_size, pos=(0,0))
        savebtn = Button(text='save', size_hint=(None, None), size=button_size, pos=(button_size[0], 0))
        loadbtn = Button(text='load', size_hint=(None, None), size=button_size, pos=(2 * button_size[0], 0))

        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)

        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        print("Clear button pressed - Resetting simulation state.")
        global first_update, total_timesteps, current_goal_index, last_distance, goal_x, goal_y
        # Reset necessary globals
        first_update = True # Force re-init on next update
        total_timesteps = 0
        current_goal_index = 0
        goal_x = goal_points[current_goal_index][0]
        goal_y = goal_points[current_goal_index][1]
        last_distance = 0 # Reset last distance
        scores.clear()
        # Clear experience buffer
        replay_buffer.storage.clear()
        replay_buffer.ptr = 0
        # Re-initialize policy? Or just let it continue learning? Better to re-initialize for true fresh start.
        # policy = TD3(state_dim, action_dim, max_action) # Re-create policy object
        print("Simulation state reset. Will re-initialize map and car.")
        # Stop/restart clock? Usually not needed, first_update handles it.

    def save(self, obj):
        print("Saving TD3 policy...")
        policy.save("td3_car_model", directory="./pytorch_models")
        try:
            plt.figure()
            plt.plot(scores)
            plt.xlabel("Timesteps")
            plt.ylabel("Reward per Step")
            plt.title("Training Rewards")
            plt.savefig("./results/training_rewards.png")
            plt.close()
            print("Reward plot saved to ./results/training_rewards.png")
        except Exception as e:
            print(f"Could not save plot: {e}")

    def load(self, obj):
        print("Loading TD3 policy...")
        policy.load("td3_car_model", directory="./pytorch_models")
        # When loading, maybe reset timesteps to allow policy usage immediately?
        # global total_timesteps
        # total_timesteps = start_timesteps


# --- Run the Application ---
if __name__ == '__main__':
    CarApp().run()