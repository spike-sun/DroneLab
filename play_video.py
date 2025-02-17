import numpy as np
import cv2
import h5py

def play_dataset(h5_path, num_envs=1, speed=2):
    wait_ms = int(50 / speed)
    with h5py.File(h5_path) as h5_file:
        total_timesteps = h5_file['episode_length'].shape[0]
        for i in range(num_envs):
            cv2.namedWindow(f'RGBD{i}', cv2.WINDOW_NORMAL)
        
        for t in range(total_timesteps):
            for i in range(num_envs):
                frame_rgb = h5_file['obs']['chaser']['rgb'][t,i]
                frame_depth = h5_file['obs']['chaser']['depth'][t,i]
                step = h5_file['episode_length'][t,i]
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frame_depth = np.clip(frame_depth, 0.0, 10.0)
                frame_depth = cv2.normalize(frame_depth, None, 0, 255, cv2.NORM_MINMAX)
                frame_depth = np.uint8(frame_depth)
                frame_depth = cv2.applyColorMap(frame_depth, cv2.COLORMAP_JET)
                frame = np.hstack((frame_bgr, frame_depth))
                cv2.putText(frame, str(step[0]), (5,214), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow(f'RGBD{i}', frame)
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    play_dataset("data/dagger/data.h5df", num_envs=1)