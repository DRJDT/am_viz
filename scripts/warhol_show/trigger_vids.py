import subprocess, os
import asyncio
import evdev
from evdev import InputDevice, ecodes
import time

###################################################################b
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

setlist = [ 'ames_harding_title',
            'tiger_balm',
            'peacock',
            'groovin',
            'unblinking_eye',
            'hummin',
            'walk_the_walk',
            'the_governors_dead',
            'cerulean_goodbye',
            'stir_my_heart_awake',
            'ames_harding_title']

tracks_dir = "/home/jd/devel/am_viz/data/track_dirs/"

num_tracks = len(setlist)

launch_delay_sec = 5.25

###################################################################

def launch_vids(track_i):

    track_name = setlist[track_i - 1]

    track_dir = tracks_dir + track_name + '/'

    print('\n')
    print("Launching Videos Track: " + str(track_i) + ' ' + track_name)
    print('\n')

    source_vid_file = track_dir + "source_vid_with_audio.mp4" # "source_vid.mp4" | "source_vid_with_audio.mp4"
    generated_vid_file = track_dir + "generated_vid.avi"
    # spectrogram_vid_file = track_dir + "spectrogram_vid.mp4"

    time.sleep(launch_delay_sec)

    subprocess.run(["killall","vlc"])
    subprocess.Popen(["vlc", source_vid_file, "--fullscreen", "--aspect-ratio", "16:9", "--no-video-title-show", "--no-loop", "--qt-fullscreen-screennumber=1"])
    subprocess.Popen(["vlc", generated_vid_file, "--fullscreen", "--aspect-ratio", "16:9", "--no-video-title-show", "--no-loop", "--qt-fullscreen-screennumber=2"])
    # subprocess.Popen(["vlc", spectrogram_vid_file, "--fullscreen", "--aspect-ratio", "16:9", "--no-video-title-show", "--no-loop", "--qt-fullscreen-screennumber=3"])


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
#         print(event)
#         print('\n')

# exit()

###################################################################

async def event_read_loop(device):

    track_i = 0

    async for ev in device.async_read_loop():

        if ev.type == ecodes.EV_KEY :
            
            if (ev.code == ecodes.KEY_B and ev.value == 1): # Play Next Track [Short Press][Right Pedal KEY_B]
            
                subprocess.run(["killall","vlc"])

                track_i = min(track_i+1,num_tracks)

                launch_vids(track_i)

            elif (ev.code == ecodes.KEY_A and ev.value == 1): # Prev Track [Short Press][Left Pedal KEY_A]

                track_i = max(track_i-1,0)
                
                subprocess.run(["killall","vlc"])

                print('\n')
                print("Track queued " + str(track_i) + ' ' + setlist[track_i])
                print('\n')

            # elif (ev.code == ecodes.KEY_A and ev.value == 2): # Skip Track [Long Press][Left Pedal KEY_A]

            #     track_i = min(track_i+1,num_tracks)

            #     print('\n')
            #     print("Track increased to " + str(track_i))
            #     print('\n')


###################################################################

loop = asyncio.get_event_loop()
loop.run_until_complete(event_read_loop(device))

