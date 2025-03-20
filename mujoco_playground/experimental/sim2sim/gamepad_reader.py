# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=line-too-long
"""Gamepad class that uses Pygame under the hood.

Adapted to use Pygame joystick interface similar to joystick_bt_sender.py.
"""
import threading
import time

import pygame
import numpy as np


# Filtering parameters similar to joystick_bt_sender.py
RC = 0.1  # Response time constant (lower = more responsive, higher = smoother)
DT = 1 / 50  # Loop time (50Hz update rate)

# Default joystick axes (works with most controllers)
J1_HORIZONTAL, J1_VERTICAL = 0, 1
J2_HORIZONTAL = 3  # Yaw rate


class FirstOrderFilter:
    """ First-order low-pass filter for smooth joystick control """
    def __init__(self, x0, rc, dt, initialized=True):
        self.x = x0
        self.dt = dt
        self.update_alpha(rc)
        self.initialized = initialized

    def update_alpha(self, rc):
        self.alpha = self.dt / (rc + self.dt)

    def update(self, x):
        if self.initialized:
            self.x = (1. - self.alpha) * self.x + self.alpha * x
        else:
            self.initialized = True
            self.x = x
        return self.x


class Gamepad:
  """Gamepad class that reads from a Pygame-compatible gamepad."""

  def __init__(
      self,
      vel_scale_x=0.4,
      vel_scale_y=0.4,
      vel_scale_rot=1.0,
      joystick_id=0,
  ):
    self._vel_scale_x = vel_scale_x
    self._vel_scale_y = vel_scale_y
    self._vel_scale_rot = vel_scale_rot
    self._joystick_id = joystick_id

    self.vx = 0.0
    self.vy = 0.0
    self.wz = 0.0
    self.is_running = True

    # Initialize filters
    self.filter_x = FirstOrderFilter(0.0, RC, DT)
    self.filter_y = FirstOrderFilter(0.0, RC, DT)
    self.filter_yaw = FirstOrderFilter(0.0, RC, DT)

    # Pygame setup
    pygame.init()
    pygame.joystick.init()
    self._joystick = None
    
    self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
    self.read_thread.start()

  def _connect_device(self):
    try:
      if pygame.joystick.get_count() == 0:
        print("No joystick detected.")
        return False
        
      self._joystick = pygame.joystick.Joystick(self._joystick_id)
      self._joystick.init()
      print(f"Connected to {self._joystick.get_name()}")
      return True
    except pygame.error as e:
      print(f"Error connecting to joystick: {e}")
      return False

  def read_loop(self):
    if not self._connect_device():
      self.is_running = False
      return

    while self.is_running:
      pygame.event.pump()  # Update joystick states
      self.update_command()
      time.sleep(DT)

    # Clean up
    if self._joystick:
      self._joystick.quit()
    pygame.quit()

  def update_command(self):
    if not self._joystick:
      return
      
    try:
      # Get raw values from joystick
      raw_y = -self._joystick.get_axis(J1_VERTICAL)  # Invert to match convention
      raw_x = self._joystick.get_axis(J1_HORIZONTAL)
      raw_yaw = self._joystick.get_axis(J2_HORIZONTAL)

      # Apply first-order filter for smoother transitions
      self.vx = self.filter_x.update(raw_y) * self._vel_scale_x
      self.vy = self.filter_y.update(raw_x) * self._vel_scale_y
      self.wz = self.filter_yaw.update(raw_yaw) * self._vel_scale_rot
    except pygame.error as e:
      print(f"Error reading joystick: {e}")

  def get_command(self):
    return np.array([self.vx, self.vy, self.wz])

  def stop(self):
    self.is_running = False


if __name__ == "__main__":
  gamepad = Gamepad()
  try:
    while True:
      command = gamepad.get_command()
      print(f"vx: {command[0]:.2f}, vy: {command[1]:.2f}, wz: {command[2]:.2f}")
      time.sleep(0.1)
  except KeyboardInterrupt:
    print("\nStopping gamepad reader...")
  finally:
    gamepad.stop()
