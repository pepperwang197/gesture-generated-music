# gesture-generated-music
### Pepper Wang & Luke Wei

The bees..... they're in your hands......

## Introduction

The goal of this project was to create an instrument to be played like a theremin, based on computer vision. Hand position would control what instrument would be played.

![Man playing a theremin](https://woodandfirestudio.com/wp-content/uploads/2022/11/dsefvd.jpg)

A theremin is an instrument, with 2 metal antennas that the performer moves their hands around. The sound depends on the proximity of each hand from the antennas - one hand controls the pitch and the other controls the volume. 

## Process

We decided to use MediaPipe's hand tracking algorithm. MediaPipe is a machine learning framework developed by Google that contains a hand landmarker model. It is able to detect 21 landmarks on a hand - the wrist, and the joints of every finger.

For our MVP, our goal was to make a working "theremin", that would play pure sine wave frequencies based on the user's hand positions. We used the tip of each hand's middle finger. If both hands are on screen, one hand's y position would control the pitch, and the other hand's x position would control the volume. 

After getting that working, we moved on to using gestures to determine instrument. We normalized a given hand's landmarks by using the wrist and the base of the middle finger as a reference to cancel out any rotation. Based on these normalized landmarks, we determined which fingertip was the highest. Each finger was mapped to a SoundFont containing audio for an instrument.

To play soundfonts, we used the Python `pysoundfont` library. Instruments like piano only have discrete notes, not a continuous range of pitches, so we used pitch bending to get between half steps. 

## Design Decisions (sinewave)

### Extracting notes from coordinates

We wanted our theremin to have a to octave range centered around middle C (C4). Meidapipe's x and y coordinates are on a scale from 0 to 1 with origin centered at the top left of the screen. In order to get a 2 octave range we would need to expand the coordinate values from 0-1 to 0-24. So, we have `note = 72 - results.hand_landmarks[0][8].y*24`. We chose 72 since it is the high end of the range (note 72 is C3, one octave above middle C), and lets the theremin play at the desired pitch range.


### Extracting frequency from notes

For the sine wave theremin we used pysinewave's `sinewave` function in to play different frequencies. Therefore, we would need to these note values into their respective frequency values. Human pitch perception is on a logarithmic scale, so the increase of pitch from 220 Hz to 440 Hz will sound the same as the increase from 220 Hz to 440 Hz. To convert the note to frequency we use the equation `freq = 440 * 2 ** ((note - 69) / 12)`440 is the frequency of A4 (note 69), 12 is the number of notes in an octave, note - 69 is how far off the note is from A4. Making this an exponential function with base 2 let's frequency double at each octave.

### Setting volume

To set the bolume for the theremin we extracted the x coordinate of the right hand and scaled that value from -60 to 0 (since volume in pysinewave is from -60 to 0). So when the right hand would be at the left edge of the screen the volume would be set to -60, and when it is on the right side of the screen it would be almost full volume.

## Design Decisions (note playing)

### Note selection

We already had a working note selection algorithm and since `pysoundfont` lets you play note themselves we just repeated the same algorithm. However just playing discrete notes created an extremely choppy sound. To combat this we integrated a pitch bending algorithm. If the value from `note`had a decimal value of less than 0.5 the pitch of the note played is bent up by a semitone, and if it is greater than 0.5 it is bent down by a semitone.

### Gesture detection

We decided to implement a very simple gesture We used media pipe's per-placed land marks to determine different gestures. To do this, we normalized each hand at 0,0 and normalized the size of each hand. We then looked at which finger's fingertip land mark had the largest y coordinate. Each Fingertip being the highest was a gesture (gesture 0 would be thumb being highest, 1 would be the index finger being the highest, etc.)

### Multiple instruments at once

Unlike the sinewave theremin, we decided to let each hand's x and y coordinate control a respective instrument's pitch and volume. At the start of the code, we initialized a synth for each instrument and only played notes from them when each respective gesture was detected from one of the hands.

## Challenges and Lessons

One challenge we faced was the lack of documentation about many of the libraries we used. For instance, when we first tried to look at mediapipe's documentation, the documentation we initially found was out of date, which led to confusion. We also found that looking some of the unofficial documentation was helpful for understanding how to properly use media pipe. Furthermore, the documentation surrounding pysinewave was incomplete, so we had to find out how to use it largely on our own.
