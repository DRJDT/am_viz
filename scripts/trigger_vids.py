import subprocess, os
import asyncio
import evdev
from evdev import InputDevice, ecodes

###################################################################

orig_vid_files = []
gen_ai_vid_files = []
spec_vid_files = []

# track_i = 1
orig_vid_files.append('outputs/tiger_balm/orig_vid.mp4')
gen_ai_vid_files.append('outputs/tiger_balm/generated_video_4.avi')
spec_vid_files.append('outputs/tiger_balm/spectrogram_video_1.mp4')

# track_i = 2
orig_vid_files.append('outputs/tiger_balm/spectrogram_video_1.mp4')
gen_ai_vid_files.append('outputs/tiger_balm/spectrogram_video_1.mp4')
spec_vid_files.append('outputs/tiger_balm/spectrogram_video_1.mp4')

NUM_TRACKS = len(orig_vid_files)

###################################################################

# Open rendered video in default system app
# subprocess.run("vlc '/home/jd/devel/am_viz/outputs/tiger_balm/tiger_balm_playlist.m3u'")

def launch_vids(track_i):

    subprocess.run(["killall","vlc"])
    subprocess.Popen(["vlc", orig_vid_files[track_i-1], "--fullscreen", "--aspect-ratio", "16:9", "--no-video-title-show", "--no-loop", "--qt-fullscreen-screennumber=1"])
    subprocess.Popen(["vlc", gen_ai_vid_files[track_i-1], "--fullscreen", "--aspect-ratio", "16:9", "--no-video-title-show", "--no-loop", "--qt-fullscreen-screennumber=2"])
    subprocess.Popen(["vlc", spec_vid_files[track_i-1], "--fullscreen", "--aspect-ratio", "16:9", "--no-video-title-show", "--no-loop", "--qt-fullscreen-screennumber=3"])


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
            elif track_i > NUM_TRACKS:
                track_i = NUM_TRACKS

            print("Launching Videos Track: " + str(track_i))
            launch_vids(track_i)

###################################################################

loop = asyncio.get_event_loop()
loop.run_until_complete(event_read_loop(device))

