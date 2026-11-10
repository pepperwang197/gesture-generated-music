# gesture-generated-music
### Pepper Wang & Luke Wei

The bees..... they're in your hands......

## Introduction

The goal of this project was to create an instrument to be played like a theremin, based on computer vision. Hand position would control what instrument would be played.

![Man playing a theremin](https://woodandfirestudio.com/wp-content/uploads/2022/11/dsefvd.jpg)

**Fig 1:** Example of man playing theremin

A theremin is an instrument, with 2 metal antennas that the performer moves their hands around. The sound depends on the proximity of each hand from the antennas - one hand controls the pitch and the other controls the volume. 

## Process

We decided to use MediaPipe's hand tracking algorithm. MediaPipe is a machine learning framework developed by Google that contains a hand landmarker model. It is able to detect 21 landmarks on a hand - the wrist, and the joints of every finger.

For our MVP, our goal was to make a working "theremin", that would play pure sine wave frequencies based on the user's hand positions. We used the tip of each hand's middle finger. If both hands are on screen, one hand's y position would control the pitch, and the other hand's x position would control the volume. 

After getting that working, we moved on to using gestures to determine instrument. We normalized a given hand's landmarks by using the wrist and the base of the middle finger as a reference to cancel out any rotation. Based on these normalized landmarks, we determined which fingertip was the highest. Each finger was mapped to a SoundFont containing audio for an instrument.

To play soundfonts, we used the Python `pysoundfont` library. Instruments like piano only have discrete notes, not a continuous range of pitches, so we used pitch bending to get between half steps. 

## Design Decisions (sinewave)

### Extracting notes from coordinates

We wanted our theremin to have a to octave range centered around middle C (C4). Meidapipe's x and y coordinates are on a scale from 0 to 1 with origin centered at the top left of the screen. In order to get a 2 octave range we would need to expand the coordinate values from 0-1 to 0-24. So, we have `note = 72 - results.hand_landmarks[0][8].y*24`. We chose 72 since it is the high end of the range (note 72 is C3, one octave above middle C), and lets the theremin play at the desired pitch range.

<img width="525" height="571" alt="Screenshot from 2026-11-10 15-34-46" src="https://github.com/user-attachments/assets/13e54ae7-cb32-4f4d-a09c-fc9d5cf3b40c" />

**Fig 2:** Visualization of landmarks place on hands by Mediapipe


### Extracting frequency from notes

For the sine wave theremin we used pysinewave's `sinewave` function in to play different frequencies. Therefore, we would need to these note values into their respective frequency values. Human pitch perception is on a logarithmic scale, so the increase of pitch from 220 Hz to 440 Hz will sound the same as the increase from 220 Hz to 440 Hz. To convert the note to frequency we use the equation `freq = 440 * 2 ** ((note - 69) / 12)`440 is the frequency of A4 (note 69), 12 is the number of notes in an octave, note - 69 is how far off the note is from A4. Making this an exponential function with base 2 let's frequency double at each octave.

<img width="810" height="468" alt="Screenshot from 2026-11-10 17-25-55" src="https://github.com/user-attachments/assets/64faba1f-d968-4c54-af7d-a9f29471bc7a" />

**Fig 3:** graph of notes with their corresponding frequencies

### Setting volume

To set the bolume for the theremin we extracted the x coordinate of the right hand and scaled that value from -60 to 0 (since volume in pysinewave is from -60 to 0). So when the right hand would be at the left edge of the screen the volume would be set to -60, and when it is on the right side of the screen it would be almost full volume.

## Design Decisions (note playing)

### Note selection

We already had a working note selection algorithm and since `pysoundfont` lets you play note themselves we just repeated the same algorithm. However just playing discrete notes created an extremely choppy sound. To combat this we integrated a pitch bending algorithm. If the value from `note`had a decimal value of less than 0.5 the pitch of the note played is bent up by a semitone, and if it is greater than 0.5 it is bent down by a semitone.

### Gesture detection

We decided to implement a very simple gesture We used media pipe's per-placed land marks to determine different gestures. To do this, we normalized each hand at 0,0 and normalized the size of each hand. We then looked at which finger's fingertip land mark had the largest y coordinate. Each Fingertip being the highest was a gesture (gesture 0 would be thumb being highest, 1 would be the index finger being the highest, etc.)

### Multiple instruments at once

Unlike the sinewave theremin, we decided to let each hand's x and y coordinate control a respective instrument's pitch and volume. At the start of the code, we initialized a synth for each instrument and only played notes from them when each respective gesture was detected from one of the hands.

## Challenges

### Sparse and outdated documentation

Several of our libraries were poorly documented. When we first looked up MediaPipe, the documentation we found was out of date, which led to confusion until we located the up-to-date documentation. We also found that looking at unofficial tutorials and community examples turned out to be a valuable supplement for understanding how to use it correctly. The documentation for `pysinewave` was even more incomplete, so we learned its behavior largely through experimentation. For example, one of the issues we had was the volume not changing when we we used the `set_volume` function. We later found out it was because we had to change the rate at which volume changes it. 

### Code organization

We didn't document or organize our code as we wrote it. File names drifted away from their contents. For example, our `sinewave.py` file ended up containing no code related to the sine wave theremin. As a result, it was hard to tell where functionality lived or what each file was responsible for, which slowed down debugging and made collaboration harder.

### Lessons

A small investment in organization at the beginning would have saved us a lot of time later. It is beneficial to focus on organizing code while we write it ex: docstrings and comments as we write code, rather than afterward. Another lesson we learned was to always git pull/push when you are starting/finishing coding. Throughout the project we largely worked synchronously together via VScode's live sharing feature. This let is bypass many of the git issues. However, this also led to us getting a little sloppy when properly pulling/pushing which led to some merge conflicts that were difficult to resolve.

## Improvements

One of the main issues with this is detecting left vs right hand. Mediapipe has a designated hand 0 and hand 1, which correspond to the left and right hands respectively. However, if Mediapipe ever loses detection of either hand, the 0 and 1  hand can swap (so left becomes 1 and right becomes 0. If we had more time we could create and implement our own left/right hand detection algorithm (for example we look at which side of the hand the thumb is on and designate left and right hand accordingly).

Another improvement we could make is the playing of different instruments. When a note is played it has an onset sound, followed by a trailing sound (think of pressing a key on a piano, it has a large initial sound with a quieter trailing sound). When we play a note via our sound font theremin, it only plays the onset sound without a trailing sound, creating a choppy sound (imagine that instead of pressing a piano key and holding it, you repeatedly press it). This can be fixed by by changing when we send notes to the synthesizer. Currently, our main loop turns the note off and back on for every video frame, so a note is restarted 30 or more times per second and we only ever hear its onset. Instead, we should track which note each hand is currently playing and send a new “note on” message only when the pitch or instrument actually changes. While the same note is held, we can keep updating its pitch bend and volume (using MIDI pitch bend and expression messages) without restarting it. This lets each note play through its full sound, including the quieter trailing portion, so it sounds like a held piano key rather than a repeatedly struck one.

## Relation to real life

Gesture recognition is widely used to control robots without physical controllers. Examples include collaborative robots in factories that respond to hand signals, surgical and assistive systems operated with touchless input, and drones or robotic arms steered by hand movements. Our mapping of mapping hand position to pitch and volume, and gesture to instrument, is the same idea as mapping hand height to a robot arm’s position.
