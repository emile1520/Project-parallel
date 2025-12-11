import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import glob
import os

def load_csv(filename):
    """Load temperature field from CSV file"""
    return np.loadtxt(filename, delimiter=',')

def create_heatmap(filename, output_image=None, show=True):
    """Create a single heatmap from a CSV file"""
    data = load_csv(filename)
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(data, cmap='hot', interpolation='bilinear', origin='lower')
    plt.colorbar(im, label='Temperature')
    plt.title(f'Temperature Field - {os.path.basename(filename)}')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    
    if output_image:
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        print(f"Saved {output_image}")
    
    if show:
        plt.show()
    else:
        plt.close()

def create_animation_from_csvs(pattern='output_step_*.csv', output_file='heat_animation.gif', fps=10, auto_scale=True):
    """Create an animation from all CSV snapshots
    
    Args:
        pattern: glob pattern for CSV files
        output_file: output filename (.gif or .mp4)
        fps: frames per second
        auto_scale: if True, scale colors dynamically for each frame; 
                   if False, use global min/max across all frames
    """
    # Get all matching files and sort them
    files = sorted(glob.glob(pattern))
    
    if len(files) == 0:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(files)} files")
    
    # Load first frame to get dimensions
    first_data = load_csv(files[0])
    
    # If not auto-scaling, find global min/max across all frames
    if not auto_scale:
        print("Computing global min/max across all frames...")
        global_min = float('inf')
        global_max = float('-inf')
        for f in files:
            data = load_csv(f)
            global_min = min(global_min, np.min(data))
            global_max = max(global_max, np.max(data))
        print(f"Global range: [{global_min:.4f}, {global_max:.4f}]")
        vmin, vmax = global_min, global_max
    else:
        # For auto-scale, we'll update these per frame
        vmin, vmax = None, None
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create initial image
    im = ax.imshow(first_data, cmap='hot', interpolation='bilinear', 
                   origin='lower', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, label='Temperature')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    title = ax.set_title(f'Temperature Field - Step 0')
    
    def update(frame):
        """Update function for animation"""
        data = load_csv(files[frame])
        im.set_array(data)
        
        # Update color limits if auto-scaling
        if auto_scale:
            im.set_clim(vmin=np.min(data), vmax=np.max(data))
        
        # Extract step number from filename
        step_num = os.path.basename(files[frame]).split('_')[-1].split('.')[0]
        title.set_text(f'Temperature Field - Step {step_num}')
        
        return [im, title]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(files), 
                                   interval=1000/fps, blit=True, repeat=True)
    
    # Save animation - try MP4 first, fall back to GIF
    print(f"Creating animation... (this may take a while)")
    try:
        if output_file.endswith('.mp4'):
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, bitrate=1800)
            anim.save(output_file, writer=writer)
        else:
            # Use PillowWriter for GIF (no ffmpeg needed)
            anim.save(output_file, writer='pillow', fps=fps)
        print(f"Animation saved to {output_file}")
    except Exception as e:
        print(f"Error creating {output_file}: {e}")
        if output_file.endswith('.mp4'):
            print("Trying GIF format instead...")
            gif_file = output_file.replace('.mp4', '.gif')
            anim.save(gif_file, writer='pillow', fps=fps)
            print(f"Animation saved to {gif_file}")
    
    plt.close()

def create_comparison_plot(files, titles=None):
    """Create a comparison plot of multiple timesteps"""
    n = len(files)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (file, ax) in enumerate(zip(files, axes)):
        data = load_csv(file)
        im = ax.imshow(data, cmap='hot', interpolation='bilinear', origin='lower')
        
        if titles and idx < len(titles):
            ax.set_title(titles[idx])
        else:
            ax.set_title(os.path.basename(file))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='T')
    
    # Hide extra subplots
    for idx in range(n, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=150, bbox_inches='tight')
    print("Saved comparison_plot.png")
    plt.show()

def analyze_convergence(filename='final_output.csv'):
    """Analyze the final temperature distribution"""
    data = load_csv(filename)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 2D heatmap
    im0 = axes[0].imshow(data, cmap='hot', interpolation='bilinear', origin='lower')
    axes[0].set_title('Final Temperature Field')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im0, ax=axes[0], label='Temperature')
    
    # Cross-section along middle row
    mid_row = data.shape[0] // 2
    axes[1].plot(data[mid_row, :], 'r-', linewidth=2)
    axes[1].set_title(f'Temperature Profile (Y = {mid_row})')
    axes[1].set_xlabel('X position')
    axes[1].set_ylabel('Temperature')
    axes[1].grid(True, alpha=0.3)
    
    # Cross-section along middle column
    mid_col = data.shape[1] // 2
    axes[2].plot(data[:, mid_col], 'b-', linewidth=2)
    axes[2].set_title(f'Temperature Profile (X = {mid_col})')
    axes[2].set_xlabel('Y position')
    axes[2].set_ylabel('Temperature')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis.png', dpi=150, bbox_inches='tight')
    print("Saved analysis.png")
    plt.show()
    
    # Print statistics
    print("\n--- Temperature Statistics ---")
    print(f"Min temperature: {np.min(data):.4f}")
    print(f"Max temperature: {np.max(data):.4f}")
    print(f"Mean temperature: {np.mean(data):.4f}")
    print(f"Std deviation: {np.std(data):.4f}")

if __name__ == "__main__":
    import sys
    
    print("=== Heat Equation Visualization Tool ===\n")
    print("Available options:")
    print("1. View final output")
    print("2. Create animation from all snapshots")
    print("3. Analyze final state")
    print("4. Compare multiple timesteps")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        if os.path.exists('final_output.csv'):
            create_heatmap('final_output.csv')
        else:
            print("Error: final_output.csv not found!")
    
    elif choice == '2':
        print("\nAnimation scaling options:")
        print("  a) Auto-scale each frame (shows heat distribution clearly)")
        print("  b) Fixed scale across all frames (shows heat dissipation)")
        scale_choice = input("Choose scaling (a/b) [default: a]: ").strip().lower()
        auto = scale_choice != 'b'
        create_animation_from_csvs(fps=10, auto_scale=auto)
    
    elif choice == '3':
        if os.path.exists('final_output.csv'):
            analyze_convergence('final_output.csv')
        else:
            print("Error: final_output.csv not found!")
    
    elif choice == '4':
        files = sorted(glob.glob('output_step_*.csv'))
        if len(files) >= 4:
            # Show first, middle, and last few frames
            selected = [files[0], files[len(files)//3], 
                       files[2*len(files)//3], files[-1]]
            create_comparison_plot(selected)
        else:
            print("Not enough snapshots for comparison")
    
    
    else:
        print("Invalid choice!")
        print("\nYou can also use this script directly:")
        print("  python visualize.py  # Interactive mode")
        print("\nOr import it in your own script:")
        print("  from visualize import create_heatmap, create_animation_from_csvs")
