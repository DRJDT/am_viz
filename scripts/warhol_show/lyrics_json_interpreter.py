import json


json_data = { 
     
    'track_name' : "Tiger Balm",
    'version' : 1.0,
    'lyrics_data' : [
    {
        'timestamp': 0.0,
        'duration' : 5.0,
        'lyrics_phrase' : "",
        'aux_prompt' : "Tiger Balm",
        'note' : "Title Card",
        'use_lyrics_as_prompt': False,
        'use_aux_prompt': True,
    },
    {
        'timestamp': 5.0,
        'duration' : 5.0,
        'lyrics_phrase' : "oh I find a, when I sleep a",
        'aux_prompt' : "rocky shore",
        'note' : "",
        'use_lyrics_as_prompt': True,
        'use_aux_prompt': True,
    },
    {
        'timestamp': 10.0,
        'duration' : 5.0,
        'lyrics_phrase' : "melodies I, sow and reap of",
        'aux_prompt' : "",
        'note' : "",
        'use_lyrics_as_prompt': True,
        'use_aux_prompt': False,
    },
    {
        'timestamp': 15.0,
        'duration' : 5.0,
        'lyrics_phrase' : "nectar sweet and the honey sweeter",
        'aux_prompt' : "honey pours",
        'note' : "",
        'use_lyrics_as_prompt': True,
        'use_aux_prompt': True,
    },
    {
        'timestamp': 20.0,
        'duration' : 5.0,
        'lyrics_phrase' : "from my lungs into the speaker",
        'aux_prompt' : "",
        'note' : "",
        'use_lyrics_as_prompt': True,
        'use_aux_prompt': False,
     },


        
],
 }





lyrics_file = '/home/jd/Software/ames_ai_viz/inputs/tiger_balm/lyrics_data.json'

# Writing to lyrics_data.json
with open(lyrics_file, "w") as outfile:
     json.dump(json_data, outfile)

     

# oh I find-a, when I sleep-a
# melodies I, sow and reap of
# nectar sweet and the honey sweeter
# from my lungs into the speaker
# tiger, I don't need yah
# i will follow, my own lead down
# to the river, as I seek-a

# tiger balm
# let me sleep with thunder
# keep me calm
# as we breathe in colors
# spread your legs
# wild and fateful flower
# bloom in the gloom
# bloom in the gloom

# freaky boy
# feed me sweet banana
# shut my eyes
# lead me through habana
# serpents tongue
# laced with fragrant nectar
# oh temptation abouds
# hypnotic, gracious and proud

# oh I find-a, when I sleep-a
# melodies I, sow and reap of
# nectar sweet and the honey sweeter
# from my lungs into the speaker
# tiger, I don't need yah
# i will follow, my own lead down
# to the river, as I seek-a

# tiger balm
# help and hide my fear
# keep me calm
# as the fates appear
# hide my face
# in your fragrant flower
# bloom in the gloom
# bloom in the gloom

# freaky gyal
# won’t you be my bassline
# shut my eyes
# as we sip the daylight
# serpent's tongue
# tells two different stories
# oh temptation abounds
# hypnotic, fatal and proud

# oh I find-a, when I sleep-a
# melodies I, sow and reap of
# nectar sweet and the honey sweeter
# from my lungs into the speaker
# tiger, I don't need yah
# i will follow, my own lead down
# to the river, as I seek-a

# oh I find-a, when I sleep-a
# melodies I, sow and reap of
# nectar sweet and the honey sweeter
# from my lungs into the speaker
# tiger, I don't need yah
# i will follow, my own lead down
# to the river, as I seek-a

    # "two": {
    #     'timestamp': 5.0,
    #     'duration' : 5.0,
    #     'lyrics_phrase' : "oh I find, when I sleep",
    #     'aux_prompt' : "rocky shore",
    #     'use_lyrics_as_prompt': True,
    #     'use_aux_prompt': True,
    #  }