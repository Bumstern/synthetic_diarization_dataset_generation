import os
import random
from typing import List, Dict
from random import shuffle, sample, choice
from pathlib import Path
import csv

from pydub import AudioSegment, effects, generators
from pyannote.audio import Pipeline
import numpy as np
from tqdm import tqdm

SILENCE_DURATION_MAXIMUM = 1000             # Максимальная длительность тишины в мс
SILENCE_DURATION_MINIMUM = 20               # Минимальная продолжительность тишины в мс (менее будут обращаться в 0)
DURATION_OF_INTERRUPTION_THRESHOLD = 100    # Максимальная продолжительность поддакивания в мс
                                            # (более будет считаться за обычный аудио-фрагмент)
OVERLAY_DURATION_MAXIMUM = 1000             # Максимальная продолжительность наложения в мс
OVERLAY_DURATION_MINIMUM = 20               # Минимальная продолжительность наложения в мс (менее не будут накладываться)
GLOBAL_FRAMERATE = 16000                    # Фреймрейт, к которому будут приведены все аудиофайлы


# TODO: Посмотреть аудио форматы с сжатием
class DatasetCreator:
    def __init__(self,
                 overlay_proba: float = 0.5,
                 random_seed: int = 0):
        self._model_pipeline = Pipeline.from_pretrained("model/config.yaml")
        assert(0 <= overlay_proba <= 1)
        self._overlay_proba = overlay_proba
        np.random.seed(random_seed)
        random.seed(random_seed)

    def pipeline(self,
                 audio_source_path: Path,
                 output_size: int,
                 noise_volume: int = -42):
        # Разделяем входные аудиодорожки на фрагменты с речью
        # и собираем их в список по каждому спикеру
        speeches = []
        for speaker_dir in os.listdir(audio_source_path):
            speaker_speech = []
            for audio_path in os.listdir(audio_source_path/speaker_dir):
                speaker_speech.append(self._speech_separator(audio_source_path/speaker_dir/audio_path))
            speeches.append(speaker_speech)

        # Создаем датасет
        speaker_names = os.listdir(audio_source_path)
        for i in tqdm(range(output_size)):
            # Выбираем двух случайных спикеров
            first_speaker_idx, second_speaker_idx = sample(range(len(speeches)), k=2)
            first_speaker_speech = choice(speeches[first_speaker_idx])
            second_speaker_speech = choice(speeches[second_speaker_idx])

            # Объединяем их в аудиотрек с разметкой
            audio, annotation = self._create_audio_track(first_speaker_speech, second_speaker_speech, noise_volume)

            # if self._
            audio = self._add_noise(audio)

            # Сохраняем аудио и разметку
            audio_name = Path(f'{speaker_names[first_speaker_idx]}__{speaker_names[second_speaker_idx]}__{str(i)}.wav')
            audio.export('dataset'/audio_name, format='wav')
            with open('annotation/' + audio_name.stem + '.csv', 'w') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=annotation[0].keys())
                csv_writer.writeheader()
                csv_writer.writerows(annotation)
        print('Done!')

    def _speech_separator(self, audio_path: Path) -> List[AudioSegment]:
        """
        Выделяет из аудиодорожки фрагменты с речью

        :param audio_path: Путь к аудиофайлу содержащего участки с речью и без
        :return: Список аудио только с речью
        """
        audio = AudioSegment.from_wav(audio_path)
        # Нормализуем аудио-файл
        audio = effects.normalize(audio)
        # Приводим к единому фреймрейту
        audio = audio.set_frame_rate(GLOBAL_FRAMERATE)
        model_segments = self._model_pipeline(audio_path)

        time_segments = []
        for speech_time in model_segments.get_timeline():
            time_segments.append([speech_time.start * 1000, speech_time.end * 1000])

        speech_audio_segments = []
        for speech_time in time_segments:
            speech_audio_segments.append(audio[speech_time[0]:speech_time[1]])
        return speech_audio_segments

    def _create_audio_track(self,
                           first_speaker: List[AudioSegment],
                           second_speaker: List[AudioSegment],
                           noise_volume: int = -42) -> (AudioSegment, List[Dict]):
        """
        Объединяет случайным образом 2 спикеров в единый аудиотрек

        :param first_speaker: Список дорожек с речью первого спикера
        :param second_speaker: Список дорожек с речью второго спикера
        :return: Аудиодорожка с перемешанной речью двух спикеров -
                 с тишиной вместо переходов | наложениями дорожек друг на друга
        """
        speaker_labels = [1] * len(first_speaker) + [2] * len(second_speaker)   # аннотация классов дорожек,
                                                                                # где 1 - первый спикер, 2 - второй
        # first_speaker, second_speaker = self.__gain_normalization(first_speaker, second_speaker)    # Нормализация по громкости
        speech_segments = first_speaker + second_speaker    # Объединенный список аудио с речью каждого спикера
        speech_segments = self.__gain_normalization(speech_segments)  # Нормализация по громкости

        # Перемешиваем будущий аудиотрек
        speech_and_class = list(zip(speech_segments, speaker_labels))
        shuffle(speech_and_class)
        speech_segments, speaker_labels = zip(*speech_and_class)

        audio_track = speech_segments[0]
        prev_label = speaker_labels[0]
        annotation = [{'speaker_label': speaker_labels[0],
                       'start_frame': 0,
                       'end_frame': int(audio_track.frame_count())}]
        for speech, label in zip(speech_segments[1:], speaker_labels[1:]):
            # Один и тот же спикер не может перебивать сам себя
            if prev_label != label:
                # Максимальное время наложения ограничено:
                # 1. Длиной накладываемой дорожки
                # 2. Не должно накладываться дольше чем продолжительность дорожки, на которую накладывают
                max_overlay_time = min(int(len(speech) * 0.2),
                                       int((annotation[-1]['end_frame'] - annotation[-1]['start_frame']) / audio_track.frame_rate * 1000))
                # 3. Не должно накладываться на предыдущего спикера
                if len(annotation) >= 2:
                    max_overlay_time = min(max_overlay_time,
                                           int((annotation[-1]['end_frame'] - annotation[-2]['end_frame']) / audio_track.frame_rate * 1000))
                # 4. Подходить по ограничениям на максимум и минимум
                if max_overlay_time > OVERLAY_DURATION_MAXIMUM:
                    max_overlay_time = OVERLAY_DURATION_MAXIMUM

                # Если слишком малое время наложения, то тогда и накладывать не нужно
                if max_overlay_time < OVERLAY_DURATION_MINIMUM:
                    audio_track, start_time, end_time = self.__append_speaker_to_track(audio_track, speech,
                                                                                       overlap=False)
                else:
                    overlap_flag = random.random() <= self._overlay_proba
                    audio_track, start_time, end_time = self.__append_speaker_to_track(audio_track, speech,
                                                                                       overlap=overlap_flag,
                                                                                       max_overlay_time=max_overlay_time)
            else:
                audio_track, start_time, end_time = self.__append_speaker_to_track(audio_track, speech, False)
            annotation.append({'speaker_label': label, 'start_frame': start_time, 'end_frame': end_time})
            prev_label = label
        return audio_track, annotation

    def __append_speaker_to_track(self,
                                  audio_track: AudioSegment,
                                  speaker: AudioSegment,
                                  overlap: bool = False,
                                  max_overlay_time: int = 200) -> (AudioSegment, int, int):
        """
        Добавляет речь спикера к аудиодорожке

        :param audio_track: Аудиодорожка
        :param speaker: Аудио с речью
        :param overlap: Накладывать ли дорожки друг на друга
        :param max_overlay_time: Максимальное время наложения (мс)
        :return:
            (Общая аудиодорожка с добавленной речью,
            Фрейм начала речи спикера,
            Фрейм конца речи спикера)
        """
        start_frame_speaker = -1
        end_frame_speaker = -1
        if overlap:
            overlap_time = random.randint(OVERLAY_DURATION_MINIMUM, max_overlay_time)
            assert(overlap_time <= len(speaker))
            start_overlap_time = len(audio_track) - overlap_time
            audio_track = audio_track.overlay(speaker, position=start_overlap_time)
            audio_track += speaker[overlap_time:]

            start_frame_speaker = int(audio_track[:start_overlap_time].frame_count())
            end_frame_speaker = int(start_frame_speaker + speaker.frame_count())
        else:
            # Генерируем продолжительность паузы
            silence_duration = int(np.random.rayleigh(scale=200))
            # Учитываем границы
            if silence_duration < SILENCE_DURATION_MINIMUM:
                silence_duration = 0
            elif silence_duration > SILENCE_DURATION_MAXIMUM:
                silence_duration = SILENCE_DURATION_MAXIMUM

            # Добавляем паузу
            if silence_duration != 0:
                audio_track += AudioSegment.silent(duration=silence_duration)
            # Вычисляем таймкод для спикера
            start_frame_speaker = int(audio_track.frame_count())
            end_frame_speaker = int(audio_track.frame_count() + speaker.frame_count())
            # Добавляем спикера
            audio_track += speaker
        assert(start_frame_speaker != -1 and end_frame_speaker != -1) # пока оставлю для дебага
        return audio_track, start_frame_speaker, end_frame_speaker

    @staticmethod
    def __gain_normalization(audio_tracks: List[AudioSegment]) -> List[AudioSegment]:
        tracks_loudness = [track.dBFS for track in audio_tracks]
        mean_loudness = np.median(tracks_loudness)    # np.median(tracks_loudness)
        target_tracks_loudness = [mean_loudness - loudness for loudness in tracks_loudness]

        normalized_tracks = []
        for (track, target_db) in zip(audio_tracks, target_tracks_loudness):
            normalized_tracks.append(track.apply_gain(target_db))
        return normalized_tracks

    @staticmethod
    def _add_noise(audio: AudioSegment) -> AudioSegment:
        noise = generators.WhiteNoise().to_audio_segment(duration=len(audio), volume=-3)
        noise = noise.set_frame_rate(GLOBAL_FRAMERATE)
        noisy_audio = audio.overlay(noise)
        return noisy_audio

    # def _add_interruptions(self,
    #                        audio_track: AudioSegment,
    #                        annotations: List[Dict],
    #                        speech_fragments: List[AudioSegment],
    #                        speech_class: List[int]) -> (AudioSegment, List[Dict]):
    #     """
    #     Добавление коротких фрагментов речи в середину монолога (поддакивание)
    #
    #     :param audio_track: Аудиодорожка
    #     :param annotations: Временные аннотации и класс отдельных спикеров
    #     :param speech_fragments: Короткие реплики (поддакивания)
    #     :param speech_class: Класс спикера реплики
    #     :return: (Аудиодорожку с поддакиваниями, Обновленную временную аннотацию)
    #     """
    #     # Выделяем фрагменты аудио, где говорит только один спикер
    #     one_speaker_only_fragments = []
    #     annotations.insert(0, {'speaker_label': None, 'start_frame': 0, 'end_frame': 0})
    #     annotations.append({
    #         'speaker_label': None,
    #         'start_frame': int(audio_track.duration_seconds * 1000),
    #         'end_frame': int(audio_track.duration_seconds * 1000)})
    #     annotations.sort(key=lambda x: x['start_frame'])
    #     for i, _ in enumerate(annotations[2:], start=2):
    #         middle = annotations[i-1]
    #         left = annotations[i-2]
    #         right = annotations[i]
    #         start_frame = None
    #         end_frame = None
    #         speaker_label = middle['speaker_label']
    #
    #         # Проверяем случаи пересечения и выделяем фрагменты
    #         if left['end_frame'] > middle['start_frame']:     # Наложение слева
    #             if right['start_frame'] < middle['end_frame']:  # Наложения с двух краев
    #                 start_frame = left['end_frame']
    #                 end_frame = right['start_frame']
    #             else:                                           # Только наложение слева
    #                 start_frame = left['end_frame']
    #                 end_frame = middle['end_frame']
    #         elif right['start_frame'] < middle['end_frame']:  # Только наложение справа
    #             start_frame = middle['start_frame']
    #             end_frame = right['start_frame']
    #         else:                                             # Наложения отсутствуют
    #             start_frame = middle['start_frame']
    #             end_frame = middle['end_frame']
    #
    #         # Если в этот промежуток времени можно вставить поддакивание, то
    #         # добавляем его в список
    #         duration = int((end_frame - start_frame) * audio_track.frame_rate * 1000)
    #         if duration > DURATION_OF_INTERRUPTION_THRESHOLD:
    #             one_speaker_only_fragments.append({
    #                 'speaker_label': speaker_label,
    #                 'start_frame': start_frame,
    #                 'end_frame': end_frame
    #             })
    #     # TODO: Реализовать вставку speech_fragments (поддакиваний)
    #     #       в промежутки времени из one_speaker_only_fragments
    #     pass

    # @staticmethod
    # def __gain_normalization(speaker1: List[AudioSegment], speaker2: List[AudioSegment]) -> (List[AudioSegment], List[AudioSegment]):
    #     speaker1_mean_db = np.mean([track.dBFS for track in speaker1])
    #     speaker2_mean_db = np.mean([track.dBFS for track in speaker2])
    #     target_db = np.median([speaker1_mean_db, speaker2_mean_db])
    #
    #     norm_speaker1 = [track.apply_gain(target_db - track.dBFS) for track in speaker1]
    #     norm_speaker2 = [track.apply_gain(target_db - track.dBFS) for track in speaker2]
    #     return norm_speaker1, norm_speaker2


