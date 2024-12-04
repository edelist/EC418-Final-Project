# import pystk


# def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=25):
#     import numpy as np
#     #this seems to initialize an object
#     action = pystk.Action()

   
 
    

#     #compute acceleration
#     action.acceleration = 0.1
    

        

    

#     return action

    




# if __name__ == '__main__':
#     from utils import PyTux
#     from argparse import ArgumentParser

#     def test_controller(args):
#         import numpy as np
#         pytux = PyTux()
#         for t in args.track:
#             steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
#             print(steps, how_far)
#         pytux.close()


#     parser = ArgumentParser()
#     parser.add_argument('track', nargs='+')
#     parser.add_argument('-v', '--verbose', action='store_true')
#     args = parser.parse_args()
    # test_controller(args)


import pystk
from utils import PyTux

def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=25):
    import numpy as np
    action = pystk.Action()

    # Compute steering based on aim_point
    action.steer = np.clip(steer_gain * aim_point[0], -1, 1)

    # Compute the difference between target and current velocity
    vel_diff = target_vel - current_vel

    # Set acceleration proportionally to the velocity difference
    action.acceleration = np.clip(vel_diff / target_vel, 0, 1)

    # Apply brake if current velocity exceeds the target velocity by more than 10%
    action.brake = current_vel > target_vel * 1.1

    # Enable drift if steering angle exceeds the skid threshold
    action.drift = abs(action.steer) > skid_thresh

    # Use nitro when accelerating and close to target velocity
    action.nitro = action.acceleration == 1 and current_vel > target_vel * 0.8

    return action

def test_controller(pytux, track, verbose=False):
    import numpy as np
    track = [track] if isinstance(track, str) else track

    for t in track:
    # Adjust max_frames based on expected completion time
        max_frames = 500
        steps, how_far = pytux.rollout(t, control, max_frames=max_frames, verbose=verbose)
        print(f"Track: {t}, Steps: {steps}, Distance: {how_far}")
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    pytux = PyTux()
    test_controller(pytux, **vars(parser.parse_args()))
    pytux.close()

# import pystk
# import math
# import sys
# '''
# Debugging
# print("sys.path:", sys.path)
# print("Package contents:", dir())
# print("sys.path:", sys.path)
# print("Module name:", __name__)
# '''
# from utils import PyTux


# #V3 OF CONTROLLER
# def control(aim_point, current_vel, depth_map=None, steer_gain=10, skid_thresh=0.6, base_target_vel=100, corner_slowdown=50):
#     import numpy as np
#     action = pystk.Action()
#     x, y = aim_point

#     # Steering logic
#     action.steer = max(-1, min(1, steer_gain * x))

#     # Dynamic speed adjustment based on steering angle
#     # Reduce target velocity as steering angle increases
#     steer_factor = abs(action.steer)
#     dynamic_target_vel = base_target_vel - steer_factor * corner_slowdown

#     # Acceleration logic
#     velocity_error = dynamic_target_vel - current_vel
#     action.acceleration = max(0, 1 - (current_vel / dynamic_target_vel))
#     action.brake = current_vel > dynamic_target_vel

#     # Drift logic
#     action.drift = abs(action.steer) > skid_thresh
#     '''    
#      # Analyze depth map if available
#     if depth_map is not None:
#         # Focus on the "danger zone" directly ahead
#         danger_zone = depth_map[int(depth_map.shape[0] * 0.6):, :]
#         min_distance = np.min(danger_zone)

#         # Adjust speed based on proximity to obstacles
#         if min_distance < 0.3:  # Example threshold for danger
#             action.brake = True
#             action.acceleration = 0
#             dynamic_target_vel = min(dynamic_target_vel, base_target_vel * 0.5)  # Slow down significantly
#     '''

#     # Nitro logic
#     if action.acceleration > 0.3 and not action.drift:
#         action.nitro = 1
#     else:
#         action.nitro = 0

#     return action


# if __name__ == '__main__':
#     from utils import PyTux
#     from argparse import ArgumentParser
    
#     def test_controller(args):
#         import numpy as np
#         pytux = PyTux()
#         for t in args.track:
#             steps, how_far = pytux.rollout(t, control, max_frames=3000, verbose=args.verbose)
#             print(steps, how_far)
#         pytux.close()
    

#     parser = ArgumentParser()
#     parser.add_argument('track', nargs='+')
#     parser.add_argument('-v', '--verbose', action='store_true')
#     args = parser.parse_args()
#     test_controller(args)


# import pystk
# import numpy as np

# def control(aim_point, current_vel, steer_gain=6.5, skid_thresh=0.25, target_vel=30):
#     action = pystk.Action()

#     action.steer = np.clip(aim_point[0] * steer_gain, -1, 1)

#     speed_error = target_vel - current_vel
#     speed_gain = 0.2
#     action.acceleration = np.clip(speed_gain * speed_error, 0, 1)
#     action.brake = speed_error < -5

#     action.drift = abs(aim_point[0]) > skid_thresh

#     action.nitro = abs(action.steer) < 0.4

#     return action




# if __name__ == '__main__':
#     from utils import PyTux
#     from argparse import ArgumentParser

#     def test_controller(args):
#         import numpy as np
#         pytux = PyTux()
#         for t in args.track:
#             steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
#             print(steps, how_far)
#         pytux.close()


#     parser = ArgumentParser()
#     parser.add_argument('track', nargs='+')
#     parser.add_argument('-v', '--verbose', action='store_true')
#     args = parser.parse_args()
#     test_controller(args)


#RESULTS:

# (intel_py310_env) (base) zanemroue2@Zanes-MacBook-Pro homework % python -m controller zengarden -v         
# Finished at t=404
# 404 0.9981356022052131

# (intel_py310_env) (base) zanemroue2@Zanes-MacBook-Pro homework % python -m controller lighthouse -v        
# Finished at t=437
# 437 0.9980535614331709

# (intel_py310_env) (base) zanemroue2@Zanes-MacBook-Pro homework % python -m controller hacienda -v          
# Finished at t=585
# 585 0.9986835805979872

# (intel_py310_env) (base) zanemroue2@Zanes-MacBook-Pro homework % python -m controller snowtuxpeak -v       
# Finished at t=568
# 568 0.999269536396353

# (intel_py310_env) (base) zanemroue2@Zanes-MacBook-Pro homework % python -m controller cornfield_crossing -v
# Finished at t=672
# 672 0.9986430505236459

# (intel_py310_env) (base) zanemroue2@Zanes-MacBook-Pro homework % python -m controller scotland -v 
# Finished at t=662
# 662 0.9988821620366168


