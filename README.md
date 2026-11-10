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

## Design Decisions

### Extracting notes from coordinates

We wanted our theremin to have a to octave range centered around middle C (C4). Meidapipe's x and y coordinates are on a scale from 0 to 1 with origin centered at the top left of the screen. In order to get a 2 octave range we would need to expand the coordinate values from 0-1 to 0-24. So, we have `note = 72 - results.hand_landmarks[0][8].y*24`. We chose 72 since it is the high end of the range (note 72 is C3, one octave above middle C), and lets the theremin play at the desired pitch range.



For the sine wave theremin we used pysinewave's `sinewave` function in to play different frequencies. 
