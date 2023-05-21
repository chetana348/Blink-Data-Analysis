# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..epochs import epochs_create
from ..signal import signal_zerocrossings


def eog_features(eog_cleaned, peaks, sampling_rate=1000):
    """**Extracts Features of EOG Eye Blinks**

    Extracts features of EOG eye blinks e.g., velocity measures, blink-amplitude-ratio (BAR),
    duration, and markers of onset and offset of each blink.

    The positive amplitude velocity ratio (pAVR) and the negative amplitude velocity ratio (nAVR).
    The positive amplitude velocity ratio is the ratio of the maximum amplitude of the blink over
    the maximum velocity (rate of change) during the blink upStroke. Similarly, the negative
    amplitude velocity ratio is the ratio of the maximum amplitude of the blink over the maximum
    velocity found in the blink downStroke. These measures have units of centiseconds and are
    indicators of fatigue.

    The blink-amplitude ratio (BAR) is the average amplitude of the signal between the blink
    leftZero and rightZero zero crossings divided by the average amplitude of the positive fraction
    of the signal “outside” the blink. BAR values in the range [5, 20]. BAR is a measure of the
    signal-to-noise ratio (SNR) of the blink to the background in a candidate signal.

    Parameters
    ----------
    eog_cleaned : Union[list, np.array, pd.Series]
        The cleaned EOG channel, extracted from :func:`.eog_clean`.
    peaks : np.array
        Vector containing the samples at which EOG-peaks occur.
    sampling_rate : int
        The sampling frequency of :func:`.eog_signal` (in Hz, i.e., samples/second).
        Defaults to 1000.

    Returns
    -------
    info : dict
        A dictionary containing information of the features of the EOG blinks, accessible with keys
        ``"Blink_LeftZeros"`` (point when eye closes), ``"Blink_RightZeros"`` (point when eye opens)
        , ``"Blink_pAVR"``, ``"Blink_nAVR"``, ``"Blink_BAR"``, and ``"Blink_Duration"`` (duration
        of each blink in seconds).

    See Also
    --------
    eog_clean, eog_findpeaks

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Get data
      eog_signal = nk.data('eog_100hz')
      eog_cleaned = nk.eog_clean(eog_signal, sampling_rate=100)
      peaks = nk.eog_findpeaks(eog_cleaned, sampling_rate=100)
      info = nk.eog_features(eog_cleaned, peaks, sampling_rate=100)

    References
    ----------
    * Kleifges, K., Bigdely-Shamlo, N., Kerick, S. E., & Robbins, K. A. (2017). BLINKER: automated
      extraction of ocular indices from EEG enabling large-scale analysis. Frontiers in
      neuroscience, 11, 12.

    """

    BARs, _, leftzeros, rightzeros, downstrokes, upstrokes = _eog_features_delineate(
        eog_cleaned, peaks, sampling_rate=sampling_rate
    )

    pAVR_list = []
    nAVR_list = []
    duration_list = []
    interblink_duration_list = []
    for i in range(len(peaks)):
        # Closing blink (pAVR)
        blink_close = upstrokes[i].Signal
        change_close = np.diff(blink_close)
        #print(f"change_close: {change_close}, length: {len(change_close)}")
        change_close = np.diff(blink_close)
        if len(change_close) == 0:
            pAVR = 0
        else:
            duration_close = len(change_close) / sampling_rate
            pAVR = (change_close.max() / duration_close) * 100
        pAVR_list.append(pAVR if len(change_close) > 0 else 0)
        # Opening blink (nAVR)
        blink_open = downstrokes[i].Signal
        change_open = np.diff(blink_open)
        if len(change_open) == 0:
            nAVR = 0
        else:
            duration_open = len(change_open) / sampling_rate
            nAVR = (change_open.max() / duration_open) * 100
        nAVR_list.append(nAVR if len(change_open) > 0 else 0)
        # Opening blink (nAVR)

        # Duration
        blink_full = np.hstack([np.array(upstrokes[i].Signal), np.array(downstrokes[i].Signal)])
        duration_full = len(blink_full) / sampling_rate  # in seconds
        duration_list.append(duration_full)
        
         # Inter-blink duration
        if i != len(peaks) - 1:
            interblink_duration = (peaks[i+1] - peaks[i]) / sampling_rate
            interblink_duration_list.append(interblink_duration)

        if len(interblink_duration_list) < len(duration_list):
            interblink_duration_list.append(np.nan)

    # Return info dictionary
    info = {
        "Blink_LeftZeros": leftzeros,
        "Blink_RightZeros": rightzeros,
        "Blink_pAVR": pAVR_list,
        "Blink_nAVR": nAVR_list,
        "Blink_BAR": BARs,
        "Blink_Duration": duration_list,
        "Interblink Duration" : interblink_duration_list
    }

    return info


# =============================================================================
# Internals
# =============================================================================


def _eog_features_delineate(eog_cleaned, candidates, sampling_rate=1000):

    # Calculate blink landmarks
    epochs = epochs_create(
        eog_cleaned,
        events=candidates,
        sampling_rate=sampling_rate,
        epochs_start=-0.5,
        epochs_end=0.5,
    )

    # max value marker
    peaks = []
    leftzeros = []
    rightzeros = []
    downstrokes = []
    upstrokes = []
    BARs = []

    for i in epochs:
        max_value = epochs[i].Signal.max()

        # Check if peak is at the end or start of epoch
        t = epochs[i].loc[epochs[i]["Signal"] == max_value].index
        if np.all(0.3 < t < 0.51):
            # Trim end of epoch
            epochs[i] = epochs[i][-0.5:0.3]
            max_value = epochs[i].Signal.max()
        if np.all(-0.51 < t < -0.3):
            # Trim start of epoch
            epochs[i] = epochs[i][-0.3:0.5]
            max_value = epochs[i].Signal.max()

        # Find position of peak
        max_frame = epochs[i]["Index"].loc[epochs[i]["Signal"] == max_value]
        max_frame = np.array(max_frame)
        if len(max_frame) > 1:
            max_frame = max_frame[0]  # If two points achieve max value, first one is blink
        else:
            max_frame = int(max_frame)

        # left and right zero markers
        crossings = signal_zerocrossings(epochs[i].Signal)
        crossings_idx = epochs[i]["Index"].iloc[crossings]
        crossings_idx = np.sort(np.append([np.array(crossings_idx)], [max_frame]))
        max_position = int(np.where(crossings_idx == max_frame)[0])

        if (max_position - 1) >= 0:  # crosses zero point
            leftzero = crossings_idx[max_position - 1]
        else:
            max_value_t = epochs[i].Signal.idxmax()
            sliced_before = epochs[i].loc[slice(max_value_t), :]
            leftzero = sliced_before["Index"].loc[
                sliced_before["Signal"] == sliced_before["Signal"].min()
            ]
            leftzero = int(np.array(leftzero))

        if (max_position + 1) < len(crossings_idx):  # crosses zero point
            rightzero = crossings_idx[max_position + 1]
        else:
            max_value_t = epochs[i].Signal.idxmax()
            sliced_before = epochs[i].loc[slice(max_value_t), :]
            sliced_after = epochs[i].tail(epochs[i].shape[0] - sliced_before.shape[0])
            rightzero = sliced_after["Index"].loc[
                sliced_after["Signal"] == sliced_after["Signal"].min()
            ]
            rightzero = np.array(rightzero)
            if rightzero.size > 0:
                rightzero = int(rightzero[0])
            else:
                rightzero = 0

        # upstroke and downstroke markers
        upstroke_idx = list(np.arange(leftzero, max_frame))
        upstroke = epochs[i].loc[epochs[i]["Index"].isin(upstroke_idx)]
        downstroke_idx = list(np.arange(max_frame, rightzero))
        downstroke = epochs[i].loc[epochs[i]["Index"].isin(downstroke_idx)]

        # left base and right base markers
        leftbase_idx = list(np.arange(epochs[i]["Index"].iloc[0], leftzero))
        leftbase_signal = epochs[i].loc[epochs[i]["Index"].isin(leftbase_idx)]
        #        leftbase_min = leftbase_signal['Signal'].min()
        #        leftbase = np.array(leftbase_signal['Index'].loc[leftbase_signal['Signal'] == leftbase_min])[0]

        rightbase_idx = list(np.arange(rightzero, epochs[i]["Index"].iloc[epochs[i].shape[0] - 1]))
        rightbase_signal = epochs[i].loc[epochs[i]["Index"].isin(rightbase_idx)]
        #        rightbase_min = rightbase_signal['Signal'].min()
        #        rightbase = np.array(rightbase_signal['Index'].loc[rightbase_signal['Signal'] == rightbase_min])[0]

        # Rejecting candidate signals with low SNR (BAR = blink-amplitude-ratio)
        inside_blink_idx = list(np.arange(leftzero, rightzero))
        inside_blink = epochs[i].loc[epochs[i]["Index"].isin(inside_blink_idx)]
        outside_blink = pd.concat([leftbase_signal, rightbase_signal], axis=0)

        BAR = inside_blink.Signal.mean() / outside_blink.Signal[outside_blink["Signal"] > 0].mean()

        # Features of all candidates
        BARs.append(BAR)
        leftzeros.append(leftzero)
        rightzeros.append(rightzero)
        downstrokes.append(downstroke)
        upstrokes.append(upstroke)

        # BAR values in the range [5, 20] usually capture blinks reasonably well
        if 3 < BAR < 50:
            peaks.append(max_frame)

    return BARs, peaks, leftzeros, rightzeros, downstrokes, upstrokes
