import subprocess, os
import evdev

orig_vid_file = "../outputs/tiger_balm/orig_vid.mp4"

gen_ai_vid_file = "../outputs/tiger_balm/generated_video_4.avi"

spectrogram_vid_file = "s../outputs/tiger_balm/spectrogram_video_1.mp4"

###################################################################

# devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
# for device in devices:
#     print(device.path, device.name, device.phys)

device = evdev.InputDevice('/dev/input/event2')
print(device)

for event in device.read_loop():
    if event.type == evdev.ecodes.EV_KEY:
        print(evdev.categorize(event))

exit()



exit()
###################################################################

# Open rendered video in default system app
# subprocess.run("vlc '/home/jd/devel/am_viz/outputs/tiger_balm/tiger_balm_playlist.m3u'")


subprocess.Popen(["vlc", orig_vid_file, "--no-one-instance", "--fullscreen", "--no-video-title-show", "--no-loop", "--qt-display-mode=2", "--qt-fullscreen-screennumber=1"])
subprocess.Popen(["vlc", gen_ai_vid_file, "--no-one-instance", "--fullscreen", "--no-video-title-show", "--no-loop", "--qt-display-mode=2", "--qt-fullscreen-screennumber=2"])
subprocess.Popen(["vlc", spectrogram_vid_file, "--no-one-instance", "--fullscreen", "--no-video-title-show", "--no-loop", "--qt-display-mode=2", "--qt-fullscreen-screennumber=3"])

