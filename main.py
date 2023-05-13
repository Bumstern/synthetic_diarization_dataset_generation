import os
from pathlib import Path
import csv
from typing import List, Dict
import wave
import datetime
from math import ceil

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from dataset_creator import DatasetCreator


def read_annotations(annotation_path: Path):
    annotation = []
    with open(annotation_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            row['start_frame'] = int(row['start_frame'])
            row['end_frame'] = int(row['end_frame'])
            annotation.append(row)
    return annotation


def plot_annotated_audio_wave(audio_path: Path, annotation: List[Dict] = None, time_step: int = None):
    colors = {'1': 'orange', '2': 'green'}  # Цвет для каждого класса
    audio = wave.open(str(audio_path), 'rb')
    framerate = audio.getframerate()
    duration_ms = int(audio.getnframes() / framerate * 1000)
    if time_step is None:
        time_step = ceil(duration_ms / (18 * 1000)) * 1000

    # Значения волновой функции аудио (байты)
    signal_wave = audio.readframes(-1)
    # Значения для оси X
    time = np.linspace(0, duration_ms, num=audio.getnframes())
    # Разные типы для разных типов кодирования
    enc_type = {
        1: np.int8,
        2: np.int16,
        4: np.int32
    }
    # Значения сигнала для оси Y
    signal_array = np.frombuffer(signal_wave,
                                 dtype=enc_type[audio.getsampwidth()])

    # Если 2 канала, то нужно проредить (выделить) сигнал только для одного канала
    # (в данном случае для левого)
    if audio.getnchannels() == 2:
        signal_array = signal_array[::2]

    # Настраиваем график
    fig = plt.figure(figsize=(20, 2))
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(left=0.05)
    fig.subplots_adjust(right=0.95)
    plt.ylabel('signal wave')
    plt.rc('axes', titlesize=10)
    plt.yticks([], [])
    # Выводим паузы
    plt.plot(time, signal_array, color='red')

    # Добавление временных подписей
    plt.tick_params(axis='x', labelsize=16)
    time_sec = np.arange(0, ceil(duration_ms / time_step) * time_step + 1, time_step)
    xticklabels = []
    for ms in time_sec:
        xticklabels.append(str(datetime.timedelta(milliseconds=round(ms))).split('.')[0])
    plt.xticks(time_sec, labels=xticklabels)
    plt.xlim([0, duration_ms])

    if annotation is not None:
        # Добавляем легенду
        file_name = audio_path.name.split('.')[0]
        label_1 = file_name.split('__')[0]
        label_2 = file_name.split('__')[1]
        plt.legend(handles=[Patch(color='orange', label=label_1), Patch(color='green', label=label_2)])

        # Теперь помечаем на нем области принадлежности к каждому спикеру
        for label in annotation:
            color = colors[label['speaker_label']]
            start_time = label['start_frame']
            end_time = label['end_frame']
            plt.plot(time[start_time:end_time], signal_array[start_time:end_time], color=color)
    plt.show()


def plot_annotated_time_frames(audio_path: Path, annotation: List[Dict], time_step: int = None):
    colors = {'1': 'orange', '2': 'green'}  # Цвет для каждого класса
    audio = wave.open(str(audio_path), 'rb')
    framerate = audio.getframerate()
    duration_ms = int(audio.getnframes() / framerate * 1000)
    if time_step is None:
        time_step = ceil(duration_ms / (18 * 1000)) * 1000
    annotation.sort(key=lambda x: x['start_frame'])

    y_main = 1          # Высота верхней временной линии (а также метка основной оси)
    y_secondary = 0     # Высота нижней временной линии (а также метка побочной оси)
    # Собираем на верхнюю временную линию все непересекающиеся фрагменты,
    # а на нижнюю все остальные
    main_annotation = [annotation[0]]
    secondary_annotation = []
    prev_axis = y_main
    for i, label in enumerate(annotation[1:], start=1):
        if label['start_frame'] < annotation[i-1]['end_frame'] and prev_axis == y_main:
            secondary_annotation.append(label)
            prev_axis = y_secondary
        else:
            main_annotation.append(label)
            prev_axis = y_main

    # Настраиваем график
    fig = plt.figure(figsize=(20, 2))
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(left=0.05)
    fig.subplots_adjust(right=0.95)
    ax = fig.add_subplot(111)
    ax.set_ylim([-1, 2])
    ax.set_yticklabels([])

    # Добавляем легенду
    file_name = audio_path.name.split('.')[0]
    label_1 = file_name.split('__')[0]
    label_2 = file_name.split('__')[1]
    ax.legend(handles=[Patch(color='orange', label=label_1), Patch(color='green', label=label_2)])

    # Добавление временных подписей
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xticks(np.arange(0, ceil(duration_ms / time_step) * time_step + 1, time_step))
    ax.set_xlim([0, duration_ms])
    time_sec = ax.get_xticks()
    xticklabels = []
    for ms in time_sec:
        xticklabels.append(str(datetime.timedelta(milliseconds=round(ms))).split('.')[0])
    ax.set_xticklabels(xticklabels)

    # Сначала верхнюю шкалу
    for label in main_annotation:
        color = colors[label['speaker_label']]
        start_time = int(label['start_frame'] / framerate * 1000)
        end_time = int(label['end_frame'] / framerate * 1000)
        ax.plot([start_time, end_time], [y_main] * 2, color=color, linewidth=10.0, solid_capstyle='butt')

    # Затем наложенную
    for label in secondary_annotation:
        color = colors[label['speaker_label']]
        start_time = int(label['start_frame'] / framerate * 1000)
        end_time = int(label['end_frame'] / framerate * 1000)
        ax.plot([start_time, end_time], [y_secondary] * 2, color=color, linewidth=10.0, solid_capstyle='butt')
    plt.show()


def plot_all_dataset():
    for audio_path, annot_path in zip(os.listdir('dataset'), os.listdir('annotation')):
        annotation = read_annotations(Path('annotation/' + annot_path))
        plot_annotated_time_frames(Path('dataset/' + audio_path), annotation, 5000)


def main():
    # plot_all_dataset()
    creator = DatasetCreator(overlay_proba=0.5, random_seed=42)
    creator.pipeline(Path('audio_test'), 1, noise_flag=True)
    # annotation = read_annotations(Path('annotation/speaker_1__speaker_3__0.csv'))
    # plot_annotated_time_frames(Path('dataset/speaker_1__speaker_3__0.wav'), annotation)
    # plot_annotated_audio_wave(Path('dataset/speaker_1__speaker_3__0.wav'), annotation)


if __name__ == '__main__':
    main()
