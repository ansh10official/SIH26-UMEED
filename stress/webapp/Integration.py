import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from EntireRec import *
from PulseSampler import *
from EDetect import *
from Common import *
from EVM import *

flag = False


def getEmotion(path, duration, PLOTSDIR):
    print("\nPROCESSING EMOTIONS")
    print("\t resizing videos.....")
    getSizes(path)
    print("\t running model.....")
    emotion_lists = getEmfromVideo(path, duration, PLOTSDIR)
    print("\tEmotion Analysis Complete!\n")
    return emotion_lists


# Plots and releases a high/medium/lower stress
def getEmotionEyebrowDetection(path, duration, fps, frame_count, PLOTSDIR):
    print("\nEYEBROW AND LIP MOVEMENT ANALYSIS")
    print("\t calculating Stress Values")
    stress_value_list, stress_level_list, fps, total = get_frame(path, duration, fps, frame_count)
    if len(stress_value_list) == 0:
        raise ValueError("No face was detected in the video.")

    xarray = np.linspace(0, duration, len(stress_value_list))
    plt.xlabel("Time (seconds)")
    plt.ylabel("Score")
    plt.axhline(y=0.75, color="red", linestyle="--", linewidth=1.1, label="High stress threshold")
    plt.plot(xarray, stress_value_list, linewidth=1.8, label="Score")
    plt.axhline(y=0.65, color="gold", linestyle="--", linewidth=1.1, label="Medium stress threshold")
    plt.title("Facial Movement Detection")
    plt.legend()
    plt.savefig(os.path.join(PLOTSDIR, 'stage2_facialmovement.png'))
    plt.clf()
    print("Lip and Eyebrow Analysis Complete!\n")
    return stress_value_list


# ----- converter
def converter(emotion_array, bpm_list, stress_value_list, duration, PLOTSDIR):
    high_stress = 1.0
    semi_high_stress = 0.8
    medium_stress = 0.5
    low_stress = 0.3
    no_stress = 0.0
    """
    ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    high stress emotions:
    anger, disgust ->high stress
    fear, surprise -> semi_high_stress
    contempt -> medium stress
    sadness -> low_stress
    happy -> no stress
    """

    if len(emotion_array) == 0 or len(bpm_list) == 0 or len(stress_value_list) == 0:
        raise ValueError("Not enough data to compute a stress score "
                         "(no frames, no heart-rate samples, or no face movement values).")

    def getHRSt(bpms):
        if 140 <= bpms:
            return high_stress
        elif bpms >= 120:
            return 0.9
        elif bpms >= 100:
            return semi_high_stress
        elif bpms >= 90:
            return medium_stress
        elif bpms >= 70:
            return low_stress
        else:
            return no_stress

    final_stress_score = []
    count = 0

    mapping = {"Anger": 0, "Disgust": 1, "Fear": 2, "Happy": 3, "Sad": 4, "Surprise": 5, "Neutral": 6}
    emotion_array = [mapping[i] for i in emotion_array]
    emotion_dict = {0: high_stress, 1: medium_stress, 2: semi_high_stress, 3: no_stress,
                    4: medium_stress, 5: high_stress, 6: medium_stress, 7: semi_high_stress}

    # never 0 — that used to crash with "modulo by zero" on short clips
    measure_skip = max(1, len(emotion_array) // len(bpm_list))

    heart_rates = []
    emotions = []
    facial_movements = []

    for i in range(len(emotion_array)):
        if i % measure_skip == 0 and count < len(bpm_list) and i < len(stress_value_list):
            total_val = 0
            total_val += emotion_dict[emotion_array[i]]
            total_val += getHRSt(bpm_list[count])
            total_val += stress_value_list[i]
            final_stress_score.append(total_val)

            heart_rates.append(bpm_list[count])
            emotions.append(emotion_array[i])
            facial_movements.append(stress_value_list[i])

            count += 1

    if not final_stress_score:
        raise ValueError("The three analyses did not overlap, so no score could be computed.")

    xarray = np.linspace(0, duration, len(final_stress_score))
    final_stress_score[0] = 0
    plt.xlabel("Time (seconds)")
    plt.ylabel("Final Stress Score")
    sns.lineplot(x=xarray, y=final_stress_score)
    plt.axhline(y=2.0, color="red", linestyle="--", linewidth=0.8, label="High Stress")
    plt.axhline(y=1.2, color="gold", linestyle="--", linewidth=0.7, label="Medium Stress")
    plt.title("Stress of Individual vs. Time")
    plt.ylim([0, 3])
    plt.legend()
    plt.savefig(os.path.join(PLOTSDIR, 'final_stressgraph.png'))
    plt.clf()

    return final_stress_score, heart_rates, emotions, facial_movements


# -------------- entire function
def getStressed(videofilename, FRAMESDIR, PLOTSDIR):
    try:
        duration, fps, frame_count = getMetrics(videofilename)

        getPicfromVideo(videofilename, FRAMESDIR)

        ### Stage 1 (EntireRec...)
        emotion_array = getEmotion(FRAMESDIR, duration, PLOTSDIR)

        ### Stage 2 (EDetect...)
        stress_value_list = getEmotionEyebrowDetection(videofilename, duration, fps, frame_count, PLOTSDIR)

        ### Stage 3 (PulseSampler...)
        if flag:
            bpm_list = getHeartRate(FRAMESDIR, duration, PLOTSDIR)
        else:
            bpm_list = getHR(FRAMESDIR, PLOTSDIR)

        print("Combining all 3 outputs into a final stress score.....")
        final_stress_score, heart_rates, emotions, facial_movements = converter(
            emotion_array, bpm_list, stress_value_list, duration, PLOTSDIR)
        print("Contactless Stress Detection Complete!")

        return final_stress_score, heart_rates, emotions, facial_movements
    finally:
        plt.close('all')   # don't let figures pile up between requests