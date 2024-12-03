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
#     test_controller(args)


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
