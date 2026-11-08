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