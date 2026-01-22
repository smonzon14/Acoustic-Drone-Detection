import numpy as np
from scipy.signal import correlate
from scipy.fft import fft, ifft
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading

def gcc_phat(sig1, sig2):
  """
  Compute GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
  between two signals.
  """
  # Compute FFT
  fft1 = fft(sig1)
  fft2 = fft(sig2)
  
  # Cross-power spectrum
  cross_spectrum = fft1 * np.conj(fft2)
  
  # Phase transform weighting
  gcc = ifft(cross_spectrum / (np.abs(cross_spectrum) + 1e-10))
  
  return np.real(gcc)


def srp_gcc_localization(signals, mic_positions, search_grid, fs=48000, c=343.0):
  """
  16-channel SRP-GCC sound source localization for 4x4 microphone array.
  """
  n_mics = signals.shape[0]
  n_samples = signals.shape[1]
  n_points = search_grid.shape[0]
  
  # Precompute GCC-PHAT for all microphone pairs
  gcc_pairs = {}
  for i in range(n_mics):
    for j in range(i + 1, n_mics):
      gcc_pairs[(i, j)] = gcc_phat(signals[i], signals[j])
  
  # SRP map
  srp_map = np.zeros(n_points)
  
  # For each candidate position
  for idx, point in enumerate(search_grid):
    srp_value = 0.0
    
    # Sum over all microphone pairs
    for i in range(n_mics):
      for j in range(i + 1, n_mics):
        # Calculate theoretical time difference of arrival (TDOA)
        dist_i = np.linalg.norm(point - mic_positions[i])
        dist_j = np.linalg.norm(point - mic_positions[j])
        tdoa = (dist_i - dist_j) / c
        
        # Convert TDOA to sample delay
        tau = int(round(tdoa * fs))
        
        # Accumulate GCC value at the predicted delay
        if -n_samples < tau < n_samples:
          srp_value += gcc_pairs[(i, j)][tau % n_samples]
    
    srp_map[idx] = srp_value
  
  # Find maximum SRP value
  max_idx = np.argmax(srp_map)
  source_position = search_grid[max_idx]
  
  return source_position, srp_map


def create_4x4_array(spacing=0.05):
  """
  Create a 4x4 square microphone array layout.
  """
  mic_positions = np.zeros((16, 3))
  idx = 0
  for i in range(4):
    for j in range(4):
      mic_positions[idx] = [i * spacing, j * spacing, 0]
      idx += 1
  
  # Center the array at origin
  mic_positions -= np.mean(mic_positions, axis=0)
  
  return mic_positions


def create_search_grid(x_range, y_range, z_range, resolution=0.05):
  """
  Create a 3D search grid.
  """
  x = np.arange(x_range[0], x_range[1], resolution)
  y = np.arange(y_range[0], y_range[1], resolution)
  z = np.arange(z_range[0], z_range[1], resolution)
  
  xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
  grid = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)
  
  return grid


class RealTimeTDoA:
  def __init__(self, mic_positions, search_grid, fs=48000, block_size=2048):
    self.mic_positions = mic_positions
    self.search_grid = search_grid
    self.fs = fs
    self.block_size = block_size
    self.n_mics = 16
    
    # Buffer for audio data
    self.audio_buffer = np.zeros((self.n_mics, block_size))
    
    # Position history
    self.position_history = deque(maxlen=50)
    self.current_position = np.array([0, 0, 0.5])
    self.srp_map = np.zeros(len(search_grid))
    
    # Lock for thread safety
    self.lock = threading.Lock()
    self.running = False
    
  def audio_callback(self, indata, frames, time, status):
    """Callback for audio stream"""
    if status:
      print(status)
    
    with self.lock:
      self.audio_buffer = indata.T  # Shape: (16, block_size)
      
      # Process localization
      try:
        pos, srp = srp_gcc_localization(
          self.audio_buffer, 
          self.mic_positions, 
          self.search_grid, 
          self.fs
        )
        self.current_position = pos
        self.srp_map = srp
        self.position_history.append(pos)
      except Exception as e:
        print(f"Localization error: {e}")
  
  def start_stream(self):
    """Start audio input stream"""
    self.running = True
    self.stream = sd.InputStream(
      device=0,
      channels=self.n_mics,
      samplerate=self.fs,
      blocksize=self.block_size,
      callback=self.audio_callback
    )
    self.stream.start()
    
  def stop_stream(self):
    """Stop audio input stream"""
    self.running = False
    if hasattr(self, 'stream'):
      self.stream.stop()
      self.stream.close()
  
  def get_current_state(self):
    """Get current position and history (thread-safe)"""
    with self.lock:
      return self.current_position.copy(), list(self.position_history), self.srp_map.copy()


def visualize_real_time(tdoa_system, search_grid):
  """
  Real-time visualization of sound source localization
  """
  fig = plt.figure(figsize=(15, 5))
  
  # 3D scatter plot for localization
  ax1 = fig.add_subplot(131, projection='3d')
  ax1.set_xlabel('X (m)')
  ax1.set_ylabel('Y (m)')
  ax1.set_zlabel('Z (m)')
  ax1.set_title('Sound Source Localization')
  
  # Plot microphone positions
  mic_pos = tdoa_system.mic_positions
  ax1.scatter(mic_pos[:, 0], mic_pos[:, 1], mic_pos[:, 2], 
              c='blue', marker='o', s=100, label='Microphones')
  
  # Initialize source position scatter
  source_scatter = ax1.scatter([], [], [], c='red', marker='*', 
                               s=300, label='Source')
  trail_scatter = ax1.scatter([], [], [], c='orange', marker='.', 
                              s=50, alpha=0.5, label='Trail')
  
  ax1.legend()
  ax1.set_xlim([-0.5, 0.5])
  ax1.set_ylim([-0.5, 0.5])
  ax1.set_zlim([0, 1.0])
  
  # 2D top view
  ax2 = fig.add_subplot(132)
  ax2.set_xlabel('X (m)')
  ax2.set_ylabel('Y (m)')
  ax2.set_title('Top View')
  ax2.scatter(mic_pos[:, 0], mic_pos[:, 1], c='blue', marker='o', s=100)
  source_2d = ax2.scatter([], [], c='red', marker='*', s=300)
  trail_2d = ax2.scatter([], [], c='orange', marker='.', s=50, alpha=0.5)
  ax2.set_xlim([-0.5, 0.5])
  ax2.set_ylim([-0.5, 0.5])
  ax2.grid(True)
  
  # SRP map visualization
  ax3 = fig.add_subplot(133)
  ax3.set_title('SRP Power Map (Max Z slice)')
  ax3.set_xlabel('X (m)')
  ax3.set_ylabel('Y (m)')
  srp_image = None
  srp_scatter = None
  
  def update(_frame):
    pos, history, srp_map = tdoa_system.get_current_state()
    
    # Update 3D scatter
    source_scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
    
    if len(history) > 0:
      history_array = np.array(history)
      trail_scatter._offsets3d = (history_array[:, 0], 
                                   history_array[:, 1], 
                                   history_array[:, 2])
      
      # Update 2D view
      source_2d.set_offsets([[pos[0], pos[1]]])
      trail_2d.set_offsets(history_array[:, :2])
    
    # Update SRP map (show max z slice)
    nonlocal srp_image, srp_scatter
    if len(srp_map) > 0:
      # Reshape SRP map to grid
      grid_shape = (len(np.unique(search_grid[:, 0])),
                    len(np.unique(search_grid[:, 1])),
                    len(np.unique(search_grid[:, 2])))
      
      if np.prod(grid_shape) == len(srp_map):
        srp_3d = srp_map.reshape(grid_shape)
        # Get max projection along z
        srp_2d = np.max(srp_3d, axis=2)
        
        if srp_image is None:
          # Create image and colorbar only once
          srp_image = ax3.imshow(srp_2d.T, origin='lower', 
                          extent=[search_grid[:, 0].min(), search_grid[:, 0].max(),
                                 search_grid[:, 1].min(), search_grid[:, 1].max()],
                          cmap='hot', aspect='auto')
          srp_scatter = ax3.scatter([pos[0]], [pos[1]], c='cyan', marker='*', s=200)
          plt.colorbar(srp_image, ax=ax3, label='SRP Value')
        else:
          # Update existing image data
          srp_image.set_data(srp_2d.T)
          srp_image.set_clim(vmin=srp_2d.min(), vmax=srp_2d.max())
          srp_scatter.set_offsets([[pos[0], pos[1]]])
    
    return source_scatter, trail_scatter, source_2d, trail_2d
  
  ani = FuncAnimation(fig, update, interval=50, blit=False)
  plt.tight_layout()
  plt.show()


# Example usage
if __name__ == "__main__":
  # Create 4x4 microphone array
  mic_positions = create_4x4_array(spacing=0.05)
  
  # Create search grid
  search_grid = create_search_grid(
    x_range=(-0.5, 0.5),
    y_range=(-0.5, 0.5),
    z_range=(0.1, 1.0),
    resolution=0.1  # Coarser grid for real-time performance
  )
  
  # Initialize real-time TDoA system
  tdoa_system = RealTimeTDoA(mic_positions, search_grid, fs=48000, block_size=4096)
  
  # Start audio stream
  print("Starting real-time localization...")
  print("Press Ctrl+C to stop")
  tdoa_system.start_stream()
  
  try:
    # Start visualization
    visualize_real_time(tdoa_system, search_grid)
  except KeyboardInterrupt:
    print("\nStopping...")
  finally:
    tdoa_system.stop_stream()
