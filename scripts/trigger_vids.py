import subprocess, os
import asyncio
import evdev
from evdev import InputDevice, ecodes
import time

###################################################################
# Setlist

# Tiger Balm
# Peacock
# Groovin
# Unblinking Eye
# Hummin
# Walk the Walk
# The Governor's Dead
# Cerulean Goodbye
# Stir My Heart Awake 

setlist = ['tiger_balm',
           'peacock',
           'groovin',
           'unblinking_eye',
           'hummin','walk_the_walk',b
           'the_governors_dead',
           'cerulean_goodbye',
           'stir_my_heart_awake']

tracks_dir = "/home/jd/devel/am_viz/inputs/track_dirs/"

num_tracks = len(setlist)

launch_delay_sec = 1.0

###################################################################

def launch_vids(track_name):

    track_dir = tracks_dir + track_name + '/'

    # source_vid_file = track_dir + "source_vid_with_audio.mp4"b

    source_vid_file = track_dir + "source_vid.mp4"
    generated_vid_file = track_dir + "generated_vid.avi"
    spectrogram_vid_file = track_dir + "spectrogram_vid.mp4"

    time.sleep(launch_delay_sec)

    subprocess.run(["killall","vlc"])
    subprocess.Popen(["vlc", source_vid_file, "--fullscreen", "--aspect-ratio", "16:9", "--no-video-title-show", "--no-loop", "--qt-fullscreen-screennumber=1"])
    subprocess.Popen(["vlc", generated_vid_file, "--fullscreen", "--aspect-ratio", "16:9", "--no-video-title-show", "--no-loop", "--qt-fullscreen-screennumber=2"])
    subprocess.Popen(["vlc", spectrogram_vid_file, "--fullscreen", "--aspect-ratio", "16:9", "--no-video-title-show", "--no-loop", "--qt-fullscreen-screennumber=3"])


###################################################################

devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
# for device in devices:
#     print(device.path, device.name, device.phys)

device = [device for device in devices if (device.name == "PCsensor FootSwitch Keyboard")][0]
# device = evdev.InputDevice('/dev/input/event2')

print(device)

# for event in device.read_loop():
#     if event.type == evdev.ecodes.EV_KEY:
#         print(evdev.categorize(event))

# exit()

###################################################################

async def event_read_loop(device):

    track_i = 0

    async for ev in device.async_read_loop():

        if ev.type == ecodes.EV_KEY and ev.value == 1:
            
            if ev.code == ecodes.KEY_B: # Next Track [Right Pedal KEY_B]
                track_i = track_i + 1

            elif ev.code == ecodes.KEY_A: # Prev Track [Left Pedal KEY_A]
                track_i = track_i - 1

            if track_i < 1:
                track_i = 1
            elif track_i > num_tracks:
                track_i = num_tracks

            track_name = setlist[track_i - 1]
            
            print("Launching Videos Track: " + str(track_i) + ' - ' + track_name)
            launch_vids(track_name)

###################################################################

loop = asyncio.get_event_loop()
loop.run_until_complete(event_read_loop(device))

