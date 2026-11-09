import tinysoundfont
import time

synth = tinysoundfont.Synth()
sfid = synth.sfload("soundfonts/trumpet_collection.sf2")
synth.program_select(0, sfid, 0, 0)
synth.start()

def position_to_note(position):
    return 72 - position * 24

# pitch bend range: 0 (-2 half steps) to 8192 to 16383 (+2 half steps)
# half a half step is 2048
# 0 - 0.5 needs to map to 0 - 2048

for i in range(200, 0, -1):
    note = position_to_note(i*0.005)
    synth.notes_off(0)
    synth.noteon(0, round(note), 70)
    diff = note%1
    print(diff)
    if diff < 0.5:
        synth.pitchbend(0, int(8192 + diff*2048*2))
        print(int(8192 + diff*1024))
    else:
        synth.pitchbend(0, int(8192 - (1-diff)*2048*2))
        print(int(8192 - (1-diff)*1024))
    time.sleep(0.1)

synth.notes_off(0)
