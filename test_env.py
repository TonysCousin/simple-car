"""Unit tester for the Racecar environment model."""

from racecar import Racecar

config = {"debug": 2}

# Case 1 - reset the env
print("\n+++++ Begin case 1")
env = Racecar(config)
obs = env.reset()
print("Case 1: reset obs = ", obs)

# Case 2 - step with invalid accel command
print("\n+++++ Begin case 2")
action = [-5.4]
try:
    env.step(action)
except Exception as e:
    print("Caught exception: ", e)

# Case 4 - step with positive accel
print("\n+++++ Begin case 4")
action = [0.04]
try:
    env.step(action)
except Exception as e:
    print("Caught exception: ", e)

# Case 5 - another step with positive accel
print("\n+++++ Begin case 5")
action = [0.045]
try:
    env.step(action)
except Exception as e:
    print("Caught exception: ", e)

# Case 6 - another step with negative accel
print("\n+++++ Begin case 6")
action = [-0.032]
try:
    env.step(action)
except Exception as e:
    print("Caught exception: ", e)
