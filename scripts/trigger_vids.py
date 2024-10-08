import subprocess, os
import asyncio
from evdev import InputDevice, ecodes

###################################################################

orig_vid_files = ["/home/jd/devel/am_viz/outputs/tiger_balm/orig_vid.mp4"]
gen_ai_vid_files = ["/home/jd/devel/am_viz/outputs/tiger_balm/generated_video_4.avi"]
spec_vid_files = ["/home/jd/devel/am_viz/outputs/tiger_balm/spectrogram_video_1.mp4"]

###################################################################

# Open rendered video in default system app
# subprocess.run("vlc '/home/jd/devel/am_viz/outputs/tiger_balm/tiger_balm_playlist.m3u'")

def launch_vids(track_i):

    subprocess.run(["killall","vlc"])
    subprocess.Popen(["vlc", orig_vid_files[track_i], "--fullscreen", "--aspect-ratio", "16:9", "--no-video-title-show", "--no-loop", "--qt-display-mode=2", "--qt-fullscreen-screennumber=1"])
    subprocess.Popen(["vlc", gen_ai_vid_files[track_i], "--fullscreen", "--aspect-ratio", "16:9", "--no-video-title-show", "--no-loop", "--qt-display-mode=2", "--qt-fullscreen-screennumber=2"])
    subprocess.Popen(["vlc", spec_vid_files[track_i], "--fullscreen", "--aspect-ratio", "16:9", "--no-video-title-show", "--no-loop", "--qt-display-mode=2", "--qt-fullscreen-screennumber=3"])


###################################################################

# devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
# for device in devices:a
#     print(device.path, device.name, device.phys)

# device = evdev.InputDevice('/dev/input/event2')
# print(device)

# for event in device.read_loop():
#     if event.type == evdev.ecodes.EV_KEY:
#         print(evdev.categorize(event))


# exit()

###################################################################

track_i = 0
launch_time = 0

device = InputDevice('/dev/input/event2')

async def event_read_loop(device):

    async for ev in device.async_read_loop():

        print(ev)
        if ev.code == ecodes.KEY_A and ev.type == 1:

            # track_i = track_i + 1
            print("Launching videos fort Track: " + str(track_i))
            launch_vids(track_i)

###################################################################

loop = asyncio.get_event_loop()
loop.run_until_complete(event_read_loop(device))

