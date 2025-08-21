"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/lift/
"""

import argparse
import json
import os
import random

import cv2
import h5py
import numpy as np

import robosuite

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    parser.add_argument("--save-video", action="store_true", help="If set, save the playback as a video")
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")

    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])

    env = robosuite.make(
        **env_info,
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    writer = None
    width, height, fps = 224, 224, 100
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = os.path.join(demo_path, "playback.mp4")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        print(f"[info] saving video to {video_path}")

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    # while True:
    for _ in range(1):
        print("Playing back random episode... (press ESC to quit)")

        # select an episode randomly
        ep = random.choice(demos)

        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.viewer.set_camera(0)

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]

        if args.use_actions:

            # load the initial state
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]

            for j, action in enumerate(actions):
                env.step(action)
                env.render()

                if args.save_video:
                    frame = env.sim.render(width=width, height=height, camera_name="agentview")
                    frame = np.flip(frame, axis=0)
                    frame = frame[..., ::-1]  # RGB -> BGR for OpenCV
                    writer.write(frame)

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()
                    if not np.all(np.equal(states[j + 1], state_playback)):
                        err = np.linalg.norm(states[j + 1] - state_playback)
                        print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")
                # if cv2.waitKey(1) & 0xFF == 27:
                #     print("[info] Viewer closed, exiting...")
                #     if writer is not None:
                #         writer.release()
                #     f.close()
                #     env.close()
                #     exit(0)

        else:

            # force the sequence of internal mujoco states one by one
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                if env.renderer == "mjviewer":
                    env.viewer.update()
                env.render()

                if args.save_video:
                    frame = env.sim.render(width=width, height=height, camera_name="agentview")
                    frame = np.flip(frame, axis=0)
                    frame = frame[..., ::-1]  # RGB -> BGR
                    writer.write(frame)

                # if cv2.waitKey(1) & 0xFF == 27:
                #     print("[info] Viewer closed, exiting...")
                #     if writer is not None:
                #         writer.release()
                #     f.close()
                #     env.close()
                #     exit(0)

    if writer is not None:
        writer.release()
    f.close()
