"""
This file holds the methods for aggregating the different base parameters of a blink
The following characteristics are found for each blinks in the following methods - 
    - start, end, width (duration)
    - duration at specific relative heights ([0.2, 0.5, 0.9])
    - location of maximum closing speed, maximum opening speed
    - blink interval to next/ previous
    - moment characteristics of blinks as distributions

The following characteristics are found per episode
    - Blink frequency per episode -> # blinks/ duration (in seconds)
"""

import pandas as pd
from scipy.signal import peak_widths, savgol_filter
from scipy.stats import moment
import numpy as np
from scipy.integrate import simps

class blink_stats:

    def perform(eog_data: list, blinks_df: pd.DataFrame):
        """
        
        """
        #blink_stats.calc_widths(eog_data, blinks_df)
        #blink_stats.calc_derivs(eog_data, blinks_df)
        #blink_stats.blink_intervals(eog_data, blinks_df, int_type='backward', colname = 'pre_interval')
        #blink_stats.blink_intervals(eog_data, blinks_df, colname='post_interval')
        #blink_stats.get_moments(eog_data, blinks_df)
        blink_stats.calc_velocities(eog_data, blinks_df)
        blink_stats.calc_accelerations(eog_data, blinks_df)
        #blink_stats.calc_auc(eog_data, blinks_df)
        

    def calc_widths(eog_data: list, blinks_df: pd.DataFrame, rel_values: list = [0, 0.2, 0.5, 0.95]):
        """
        Calculate the array-width of specified blinks (in `blinks_df`) at relative heights from `rel_values`. \n
        This function utilizes `scipy.signal.peak_widths`. \n
        Arguments:
            - `eog_data`: general EOG readings of the sample
            - `blinks_df`: `neurokit2`-generated blinks from eog data as a `pd.DataFrame`
            - `rel_values`: list of relative heights at width width must be calculated. Here `0` equates to 
            blink width while `0.5` refers to the width of blinks at half height. \n
        
        Returns: `None` \n
        Columns are added to the `blink_df` DataFrame, with the `"{rel_height}width"` naming convention
        """
        for val in rel_values:
            width, width_vals, b, c = peak_widths(eog_data, blinks_df["Peaks"], rel_height = 1 - val)
            blinks_df[f"{val}width"] = width
    
    def calc_derivs(eog_data: list, blinks_df: pd.DataFrame, window_length: int = 3, polyorder: int =2):
        """
        Perform the derivative calculation for blinks through `scipy.signal.savgol_filter` \n
        The respective columns added in this function are:
            - `spmax_close`: array index of max derivative while closing of eyelids
            - `spmin_open`: array index of min derivative while opening of eyelids
            - Something with second derivative?
        """

        first_deriv = savgol_filter(eog_data, window_length=window_length, polyorder=polyorder, deriv = 1)
        second_deriv = savgol_filter(eog_data, window_length=window_length, polyorder=polyorder, deriv = 2)

        spmax = []
        spmin = []
        for count, blink in blinks_df.iterrows():
            start = blink["Onsets"]
            end = blink["Offsets"]
            if np.isnan(start) or np.isnan(end):
                spmax.append(np.nan)
                spmin.append(np.nan)
                continue
            start = int(start)
            end = int(end)

            first_deriv_part = first_deriv[start: end]
            sec_deriv_part = second_deriv[start:end]
            spmax.append(np.where(first_deriv_part == np.max(first_deriv_part))[0][0])
            spmin.append(np.where(first_deriv_part == np.min(first_deriv_part))[0][0])

        blinks_df["spmax_close"] = pd.Series(spmax, dtype = "Int64")
        blinks_df['spmin_open'] = pd.Series(spmin, dtype = "Int64")
    
    def calc_velocities(eog_data: list, blinks_df: pd.DataFrame, window_length: int = 3, polyorder: int = 2):
        """
        Calculate the velocity of opening and closing of eyelids during blinks using `scipy.signal.savgol_filter`.
        The respective columns added in this function are:
            - `closing_velocity`: velocity of closing eyelids
            - `opening_velocity`: velocity of opening eyelids
        """
        first_deriv = savgol_filter(eog_data, window_length=window_length, polyorder=polyorder, deriv=1)

        closing_velocities = []
        opening_velocities = []
        for _, blink in blinks_df.iterrows():
            start = int(blink["Onsets"])
            end = int(blink["Offsets"])
            if np.isnan(start) or np.isnan(end):
                closing_velocities.append(np.nan)
                opening_velocities.append(np.nan)
                continue

            first_deriv_part = first_deriv[start:end]
            peak_max = np.where(first_deriv_part == np.max(first_deriv_part))[0][0]
            peak_min = np.where(first_deriv_part == np.min(first_deriv_part))[0][0]

            closing_velocities.append(np.abs(first_deriv_part[0] - first_deriv_part[peak_max]))
            opening_velocities.append(np.abs(first_deriv_part[peak_min] - first_deriv_part[-1]))

        blinks_df["closing_velocity"] = pd.Series(closing_velocities, dtype="float32")
        blinks_df["opening_velocity"] = pd.Series(opening_velocities, dtype="float32")
    
    def calc_accelerations(eog_data: list, blinks_df: pd.DataFrame, window_length: int = 3, polyorder: int = 2):
        """
        Calculate the acceleration of opening and closing of eyelids during blinks using `scipy.signal.savgol_filter`.
        The respective columns added in this function are:
            - `closing_acceleration`: acceleration of closing eyelids
            - `opening_acceleration`: acceleration of opening eyelids
        """
        first_deriv = savgol_filter(eog_data, window_length=window_length, polyorder=polyorder, deriv=1)
        second_deriv = savgol_filter(eog_data, window_length=window_length, polyorder=polyorder, deriv=2)

        closing_accelerations = []
        opening_accelerations = []
        for _, blink in blinks_df.iterrows():
            start = int(blink["Onsets"])
            end = int(blink["Offsets"])
            if np.isnan(start) or np.isnan(end):
                closing_accelerations.append(np.nan)
                opening_accelerations.append(np.nan)
                continue

            first_deriv_part = first_deriv[start:end]
            sec_deriv_part = second_deriv[start:end]
            peak_max = np.where(first_deriv_part == np.max(first_deriv_part))[0][0]
            peak_min = np.where(first_deriv_part == np.min(first_deriv_part))[0][0]

            closing_accelerations.append(np.abs(sec_deriv_part[0] - sec_deriv_part[peak_max]))
            opening_accelerations.append(np.abs(sec_deriv_part[peak_min] - sec_deriv_part[-1]))

        blinks_df["closing_acceleration"] = pd.Series(closing_accelerations, dtype="float32")
        blinks_df["opening_acceleration"] = pd.Series(opening_accelerations, dtype="float32")
        
    def calc_auc(eog_data: list, blinks_df: pd.DataFrame):
        """
        Calculate the area under the curve (AUC) for each blink using `scipy.integrate.simps`.
        The respective column added in this function is:
            - `auc`: area under the curve for each blink
        """
        auc_values = []
        for _, blink in blinks_df.iterrows():
            start = int(blink["Onsets"])
            end = int(blink["Offsets"])
            if np.isnan(start) or np.isnan(end):
                auc_values.append(np.nan)
                continue

            auc_values.append(simps(eog_data[start:end]))

        blinks_df["auc"] = pd.Series(auc_values, dtype="float32")
   

    
    def blink_intervals(eog_data: list, blinks_df: pd.DataFrame, int_type = 'forward', colname = "pre_interval"):
        """
        Track the array distance between blinks. \n
        Option `int_type` list the interval type -> `{'forward', 'backward'}`
        """
        assert int_type == 'forward' or int_type == 'backward', f"Incorrect interval type {int_type}"
        blink_intervals = []

        for count, blink in blinks_df.iterrows():
            if int_type == 'forward':
                if count + 1 == len(blinks_df):
                    blink_intervals.append(np.nan)
                    continue
                
                blink_intervals.append(blinks_df['Onsets'][count] - blinks_df['Offsets'][count])


            elif int_type == 'backward':
                if count == 0:
                    blink_intervals.append(np.nan)
                    continue
                if count > 1:
                    blink_intervals.append(blinks_df['Onsets'][count] - blinks_df['Offsets'][count - 1])
                else:
                    blink_intervals.append(np.nan)
        blinks_df[colname] = pd.Series(blink_intervals, dtype = "Int64")
    
    def get_moments(eog_cleaned: list, blinks_df: pd.DataFrame, moments: list = [1, 2, 3, 4]):
        return_vals = []
        for count, blink in blinks_df.iterrows():
            start = blink["Onsets"]
            end = blink["Offsets"]
            if np.isnan(start) or np.isnan(end):
                return_vals.append([np.nan] * len(moments))
                continue
            start = int(start)
            end = int(end)
            blink_range = eog_cleaned[start: end]
            curr_blink_moments = []
            for val in moments:
                val_moment = moment(blink_range, moment = val, nan_policy = 'omit')
                curr_blink_moments.append(val_moment)
            return_vals.append(curr_blink_moments)
        new_df = pd.DataFrame(return_vals, columns = [f"moment{val}" for val in moments])
        for i in new_df.columns:
            blinks_df[i] = new_df[i]




def label_doubleblinks(blinks_df: pd.DataFrame):
    """
    Label blinks based on whether they're double blinks

    Labeling schema:
        - `0`: single blink
        - `1`: first blink in double blink pair
        - `2`: second blink in double blink pair
    
    Arguments: 
        - `blinks_df`: `neurokit2`-generated blinks from eog data as a `pd.DataFrame`
    
    Returns:
        - `labels`: list of corresponding labels
    """
    labels = []
    for count, blink in blinks_df.iterrows():
        if not pd.isna(blink["post_interval"] ) and blink["post_interval"] == 0:
            labels.append(1)
        elif not pd.isna(blink["pre_interval"]) and blink["pre_interval"] == 0:
            labels.append(2)
        else:
            labels.append(0)
    return np.array(labels)

def gen_blinktable(blink_eog: list, blink_lims: list[list[int]]) -> pd.DataFrame:
    """
    Given a list of blink slices, this method generates the starting 
    `neurokit`-like blink `pd.DataFrame`. It provides the following columns:
        - `"Onsets"`: Start of blink (eog array index)
        - `"Offsets"`: End of blink (eog array index)
        - `"Peak"`: blink peak (eog array index)
        - `"rise_height"`: Height of blink from start position. 
        - `"fall_height"`: Height of blink to end position.\n
    Height is calculated with the provided units in `blink_eog`. \n
    Arguments:
        - `blink_eog`: List of eog readings for the event
        - `blink_lims`: list of integer tuples containing `[start, end]` slices
        for blinks \n

    Returns: 
        - `blinks_df`: `pandas.DataFrame` table of blinks with information
    """
    blinks_arr = []
    for blink in blink_lims:
        start = int(blink[0])
        end = int(blink[1])
        blink_range = blink_eog[start:end]

        peak = np.where(blink_range == np.max(blink_range))[0][0]
        r_height = blink_range[peak] - blink_eog[start]
        f_height = blink_range[peak] - blink_eog[end]
        blinks_arr.append([start, end, peak + start, r_height, f_height])
    blinks_df = pd.DataFrame(blinks_arr, columns = ['Onsets', "Offsets", "Peaks", "rise_height", "fall_height"])
    return blinks_df

