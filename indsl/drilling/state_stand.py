# Copyright 2026 Cognite AS

from enum import IntEnum
import numpy as np
import pandas as pd
from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types

class StateStand(IntEnum):
    """Drilling state stand codes."""
    ST_ABSENT = 0  # One or more required channels absent
    ST_DRILLING = 1  # On-bottom drilling
    ST_OFFBOT = 2  # Offbottom during drilling of a stand
    ST_WIPE = 3  # Wiper trip
    ST_RIH = 4  # Run in hole
    ST_POOH = 5  # Pull out of hole
    ST_CONNECTION = 6  # Drill string is in slips
    ST_WEIGHTSLIP = 7  # Drill string up offbottom until pipe in slips
    ST_SLIPWEIGHT = 8  # Drill string offbottom to resume drilling

class StateMode(IntEnum):
    """Drilling state mode codes."""
    MD_ABSENT = 0  # Not enough data
    MD_DRILLROTATE = 1  # Rotate drilling
    MD_DRILLSLIDE = 2  # Slide drilling
    MD_SLIPS = 3  # Slips
    MD_REAM = 4  # Ream In
    MD_TRIPPUMP = 5  # Moving in with pumping but no rotation
    MD_TRIPROTATE = 6  # Moving in with rotation but no pumping
    MD_TRIP = 7  # Moving in without pumping or rotation
    MD_REAMOUT = 8  # Back reaming
    MD_PUMPOUT = 9  # Moving out with pumping but not rotating
    MD_ROTATEOUT = 10  # Moving out with rotation but no pumping
    MD_TRIPOUT = 11  # Moving out without pumping or rotation
    MD_PUMPROTATE = 12  # Static with pumping and rotation
    MD_PUMP = 13  # Static with pumping but not rotation
    MD_ROTATE = 14  # Static with rotation but no pumping
    MD_STATIC = 15  # Static

@check_types
def state_stand(
    time: pd.Series,
    hkld: pd.Series,
    dmd: pd.Series,
    dbit: pd.Series,
    bpos: pd.Series,
    rpm: pd.Series,
    tors: pd.Series,
    spp: pd.Series,
    flow: pd.Series,
) -> pd.DataFrame:
    r"""Drilling state stand detection.

    Detects drilling operational states at the stand level and calculates block velocity based on multiple drilling parameters.
    The function classifies drilling activities into different state modes and computes the block velocity
    from block position changes over time.

    Block velocity is calculated as the time derivative of block position:

    .. math::
        \mathrm{bvel} = \frac{d(\mathrm{bpos})}{dt}

    Where:
    - :math:`\mathrm{bpos}` is the block position [m]
    - :math:`t` is time [ms]
    - :math:`\mathrm{bvel}` is the block velocity [:math:`\mathrm{m/s}`]

    The state_stand is an integer code representing drilling operational states at the stand level:
    - 0: ST_ABSENT - One or more required channels absent
    - 1: ST_DRILLING - On-bottom drilling
    - 2: ST_OFFBOT - Offbottom during drilling of a stand
    - 3: ST_WIPE - Wiper trip
    - 4: ST_RIH - Run in hole
    - 5: ST_POOH - Pull out of hole
    - 6: ST_CONNECTION - Drill string is in slips
    - 7: ST_WEIGHTSLIP - Drill string up offbottom until pipe in slips
    - 8: ST_SLIPWEIGHT - Drill string offbottom to resume drilling

    The state_mode is an integer code representing detailed drilling operational modes:
    - 0: MD_ABSENT - Not enough data
    - 1: MD_DRILLROTATE - Rotate drilling
    - 2: MD_DRILLSLIDE - Slide drilling
    - 3: MD_SLIPS - Slips
    - 4: MD_REAM - Ream In
    - 5: MD_TRIPPUMP - Moving in with pumping but no rotation
    - 6: MD_TRIPROTATE - Moving in with rotation but no pumping
    - 7: MD_TRIP - Moving in without pumping or rotation
    - 8: MD_REAMOUT - Back reaming
    - 9: MD_PUMPOUT - Moving out with pumping but not rotating
    - 10: MD_ROTATEOUT - Moving out with rotation but no pumping
    - 11: MD_TRIPOUT - Moving out without pumping or rotation
    - 12: MD_PUMPROTATE - Static with pumping and rotation
    - 13: MD_PUMP - Static with pumping but not rotation
    - 14: MD_ROTATE - Static with rotation but no pumping
    - 15: MD_STATIC - Static

    Args:
        time: Time [ms].
            Time series in milliseconds since 1970-01-01 (Unix epoch). Used for temporal analysis
            and time-based calculations.
        hkld: HookLoad [kN].
            Time series with hookload values. Used to detect slips (very low HookLoad) and drilling activity.
        dmd: HoleDepth [m].
            Time series with hole depth values. Used to detect drilling progress and connection intervals.
        dbit: BitDepth [m].
            Time series with bit depth values. Used to detect when bit is on-bottom vs off-bottom.
        bpos: BlockPosition [m].
            Time series with block position values. Used to detect pipe movement during connections.
        rpm: Rotary Speed [rpm].
            Time series with rotary speed values. Used to classify rotating vs non-rotating states.
        tors: SurfaceTorque [kN.m].
            Time series with surface torque values. Used to detect rotation and drilling activity.
        spp: StandpipePressure [kPa].
            Time series with standpipe pressure values. Used to detect pumping activity.
        flow: FlowIn [:math:`\mathrm{L/min}`].
            Time series with flow-in values. Used to detect pumping activity.

    Returns:
        pandas.DataFrame: Drilling state results.
            DataFrame with three columns:
            - `bvel`: Block velocity [:math:`\mathrm{m/s}`] calculated from block position changes over time.
                Returns NaN if insufficient data points (< 2) are provided.
            - `state_stand`: Integer code representing the drilling operational state at stand level (0-8).
                Returns ST_ABSENT (0) if insufficient data points are provided.
            - `state_mode`: Integer code representing the detailed drilling operational mode (0-15).
                Returns MD_ABSENT (0) if insufficient data points are provided.
    """
    # Convert time from milliseconds to datetime index
    # Time input values are in milliseconds since epoch (Unix epoch)
    # Convert to datetime index for proper time-based calculations
    if isinstance(time.index, pd.DatetimeIndex):
        # If already has datetime index, assume values are milliseconds and create new index
        time_ms = time.values
        time_index = pd.to_datetime(time_ms, unit="ms")
    else:
        # Convert time values (ms) to datetime index
        time_ms = time.values
        time_index = pd.to_datetime(time_ms, unit="ms")
    
    # Create time series with datetime index
    time_series = pd.Series(time_ms, index=time_index)

    # Hardcoded thresholds
    hkld_slip_threshold = 50.0  # kN
    rpm_rotation_threshold = 10.0  # rpm
    tors_rotation_threshold = 1.0  # kN.m
    spp_pumping_threshold = 100.0  # kPa
    flow_pumping_threshold = 50.0  # L/min

    # Align all time series (including time series for consistency)
    all_series = [time_series, hkld, dmd, dbit, bpos, rpm, tors, spp, flow]
    aligned_series = auto_align(all_series, False)  # type: ignore
    time_aligned, hkld_aligned, dmd_aligned, dbit_aligned, bpos_aligned, rpm_aligned, tors_aligned, spp_aligned, flow_aligned = aligned_series

    # Check if we have sufficient data
    if len(time_aligned) < 2:
        # Return NaN values for all outputs if insufficient data
        result = pd.DataFrame(
            {
                "bvel": pd.Series([np.nan] * len(time_aligned), index=time_aligned.index),
                "state_stand": pd.Series([StateStand.ST_ABSENT] * len(time_aligned), index=time_aligned.index, dtype=np.int64),
                "state_mode": pd.Series([StateMode.MD_ABSENT] * len(time_aligned), index=time_aligned.index, dtype=np.int64),
            },
            index=time_aligned.index,
        )
        return result

    # Calculate block velocity from block position and time
    # Use numpy gradient for numerical differentiation
    # Convert time to seconds for velocity calculation
    time_seconds = time_aligned.index.view(np.int64) / 1e9  # Convert nanoseconds to seconds
    bpos_values = bpos_aligned.values

    # Calculate velocity: dv = dp/dt
    # Use central difference for interior points, forward/backward for boundaries
    bvel_values = np.gradient(bpos_values, time_seconds)

    # Initialize state arrays
    state_stand = np.zeros(len(time_aligned), dtype=np.int64)
    state_mode = np.zeros(len(time_aligned), dtype=np.int64)

    # Get aligned values
    hkld_vals = hkld_aligned.values
    dmd_vals = dmd_aligned.values
    dbit_vals = dbit_aligned.values
    bpos_vals = bpos_aligned.values
    rpm_vals = rpm_aligned.values
    tors_vals = tors_aligned.values
    spp_vals = spp_aligned.values
    flow_vals = flow_aligned.values

    # Check for missing data (ST_ABSENT = 0, MD_ABSENT = 0)
    has_missing = (
        np.isnan(hkld_vals) | np.isnan(dmd_vals) | np.isnan(dbit_vals) | np.isnan(bpos_vals) |
        np.isnan(rpm_vals) | np.isnan(tors_vals) | np.isnan(spp_vals) | np.isnan(flow_vals)
    )
    state_stand[has_missing] = StateStand.ST_ABSENT
    state_mode[has_missing] = StateMode.MD_ABSENT

    # Only process non-missing data
    valid_mask = ~has_missing
    if not np.any(valid_mask):
        # All data is missing
        result = pd.DataFrame(
            {
                "bvel": bvel_values,
                "state_stand": state_stand,
                "state_mode": state_mode,
            },
            index=time_aligned.index,
        )
        return result

    # Calculate on-bottom status (bit depth close to hole depth)
    on_bottom = np.abs(dmd_vals - dbit_vals) < 0.2  # Within 0.2m considered on-bottom

    # Detect rotation (either RPM or torque above threshold)
    is_rotating = (rpm_vals > rpm_rotation_threshold) | (tors_vals > tors_rotation_threshold)

    # Detect pumping (either standpipe pressure or flow above threshold)
    is_pumping = (spp_vals > spp_pumping_threshold) | (flow_vals > flow_pumping_threshold)

    # Detect slips (low hookload)
    is_slips = hkld_vals < hkld_slip_threshold

    # Detect block movement
    # Note: When bpos decreases, block moves up (tripping out) - negative velocity
    #       When bpos increases, block moves down (tripping in) - positive velocity
    bvel_abs = np.abs(bvel_values)
    is_moving_up = bvel_values < -0.01  # Moving up (negative velocity, bpos decreasing)
    is_moving_down = bvel_values > 0.01  # Moving down (positive velocity, bpos increasing)
    is_stationary = bvel_abs < 0.01  # Stationary

    # Apply valid mask to all conditions
    valid_on_bottom = on_bottom & valid_mask
    valid_is_rotating = is_rotating & valid_mask
    valid_is_pumping = is_pumping & valid_mask
    valid_is_slips = is_slips & valid_mask
    valid_is_moving_up = is_moving_up & valid_mask
    valid_is_moving_down = is_moving_down & valid_mask
    valid_is_stationary = is_stationary & valid_mask

    # Classify state_mode (detailed modes) - order matters (more specific first)
    # MD_ABSENT already set for missing data

    # MD_SLIPS - Slips (highest priority after absent)
    state_mode[valid_is_slips] = StateMode.MD_SLIPS

    # MD_DRILLROTATE - Rotate drilling (on-bottom, rotating, pumping, stationary)
    # Must check before MD_PUMPROTATE to prioritize drilling over static
    drilling_rotate_mask = valid_on_bottom & valid_is_rotating & valid_is_pumping & valid_is_stationary & ~valid_is_slips
    state_mode[drilling_rotate_mask] = StateMode.MD_DRILLROTATE

    # MD_DRILLSLIDE - Slide drilling (on-bottom, not rotating, pumping, stationary)
    # Must check before MD_PUMP to prioritize drilling over static
    drilling_slide_mask = valid_on_bottom & ~valid_is_rotating & valid_is_pumping & valid_is_stationary & ~valid_is_slips
    state_mode[drilling_slide_mask] = StateMode.MD_DRILLSLIDE

    # MD_REAM - Ream In (moving down, rotating, pumping)
    ream_mask = valid_is_moving_down & valid_is_rotating & valid_is_pumping & ~valid_is_slips & (state_mode == StateMode.MD_ABSENT)
    state_mode[ream_mask] = StateMode.MD_REAM

    # MD_REAMOUT - Back reaming (moving up, rotating, pumping)
    reamout_mask = valid_is_moving_up & valid_is_rotating & valid_is_pumping & ~valid_is_slips & (state_mode == StateMode.MD_ABSENT)
    state_mode[reamout_mask] = StateMode.MD_REAMOUT

    # MD_TRIPPUMP - Moving in with pumping but no rotation
    trippump_mask = valid_is_moving_down & ~valid_is_rotating & valid_is_pumping & ~valid_is_slips & (state_mode == StateMode.MD_ABSENT)
    state_mode[trippump_mask] = StateMode.MD_TRIPPUMP

    # MD_TRIPROTATE - Moving in with rotation but no pumping
    triprotate_mask = valid_is_moving_down & valid_is_rotating & ~valid_is_pumping & ~valid_is_slips & (state_mode == StateMode.MD_ABSENT)
    state_mode[triprotate_mask] = StateMode.MD_TRIPROTATE

    # MD_TRIP - Moving in without pumping or rotation
    trip_mask = valid_is_moving_down & ~valid_is_rotating & ~valid_is_pumping & ~valid_is_slips & (state_mode == StateMode.MD_ABSENT)
    state_mode[trip_mask] = StateMode.MD_TRIP

    # MD_PUMPOUT - Moving out with pumping but not rotating
    pumpout_mask = valid_is_moving_up & ~valid_is_rotating & valid_is_pumping & ~valid_is_slips & (state_mode == StateMode.MD_ABSENT)
    state_mode[pumpout_mask] = StateMode.MD_PUMPOUT

    # MD_ROTATEOUT - Moving out with rotation but no pumping
    rotateout_mask = valid_is_moving_up & valid_is_rotating & ~valid_is_pumping & ~valid_is_slips & (state_mode == StateMode.MD_ABSENT)
    state_mode[rotateout_mask] = StateMode.MD_ROTATEOUT

    # MD_TRIPOUT - Moving out without pumping or rotation
    tripout_mask = valid_is_moving_up & ~valid_is_rotating & ~valid_is_pumping & ~valid_is_slips & (state_mode == StateMode.MD_ABSENT)
    state_mode[tripout_mask] = StateMode.MD_TRIPOUT

    # MD_PUMPROTATE - Static with pumping and rotation (off-bottom)
    pumprotate_mask = valid_is_stationary & valid_is_rotating & valid_is_pumping & ~valid_on_bottom & ~valid_is_slips & (state_mode == StateMode.MD_ABSENT)
    state_mode[pumprotate_mask] = StateMode.MD_PUMPROTATE

    # MD_PUMP - Static with pumping but not rotation (off-bottom)
    pump_mask = valid_is_stationary & ~valid_is_rotating & valid_is_pumping & ~valid_on_bottom & ~valid_is_slips & (state_mode == StateMode.MD_ABSENT)
    state_mode[pump_mask] = StateMode.MD_PUMP

    # MD_ROTATE - Static with rotation but no pumping
    rotate_mask = valid_is_stationary & valid_is_rotating & ~valid_is_pumping & ~valid_is_slips & (state_mode == StateMode.MD_ABSENT)
    state_mode[rotate_mask] = StateMode.MD_ROTATE

    # MD_STATIC - Static (no movement, no rotation, no pumping)
    static_mask = valid_is_stationary & ~valid_is_rotating & ~valid_is_pumping & ~valid_is_slips & (state_mode == StateMode.MD_ABSENT)
    state_mode[static_mask] = StateMode.MD_STATIC

    # Classify state_stand (stand-level states)
    # ST_ABSENT already set for missing data

    # ST_CONNECTION - Drill string is in slips
    state_stand[valid_is_slips] = StateStand.ST_CONNECTION

    # ST_DRILLING - On-bottom drilling (on-bottom, rotating or not, pumping, stationary)
    # Includes both rotate drilling and slide drilling
    drilling_stand_mask = (drilling_rotate_mask | drilling_slide_mask)
    state_stand[drilling_stand_mask] = StateStand.ST_DRILLING

    # ST_OFFBOT - Offbottom during drilling of a stand (off-bottom, rotating, pumping, stationary)
    # This should match MD_PUMPROTATE
    offbot_mask = ~valid_on_bottom & valid_is_rotating & valid_is_pumping & valid_is_stationary & ~valid_is_slips
    state_stand[offbot_mask] = StateStand.ST_OFFBOT

    # ST_WIPE - Wiper trip (moving up, rotating, pumping) - similar to back reaming
    state_stand[reamout_mask] = StateStand.ST_WIPE

    # ST_RIH - Run in hole (moving down, not rotating)
    rih_mask = valid_is_moving_down & ~valid_is_rotating & ~valid_is_slips
    state_stand[rih_mask] = StateStand.ST_RIH

    # ST_POOH - Pull out of hole (moving up, not rotating)
    pooh_mask = valid_is_moving_up & ~valid_is_rotating & ~valid_is_slips
    state_stand[pooh_mask] = StateStand.ST_POOH

    # ST_WEIGHTSLIP - Drill string up offbottom until pipe in slips
    # This is when moving up towards slips (approaching slips while moving up)
    weightslip_mask = valid_is_moving_up & ~valid_is_rotating & ~valid_is_pumping & ~valid_is_slips & (state_stand == StateStand.ST_ABSENT)
    state_stand[weightslip_mask] = StateStand.ST_WEIGHTSLIP

    # ST_SLIPWEIGHT - Drill string offbottom to resume drilling
    # This is when moving down from slips to resume drilling
    slipweight_mask = valid_is_moving_down & ~valid_is_rotating & ~valid_is_pumping & ~valid_is_slips & (state_stand == StateStand.ST_ABSENT)
    state_stand[slipweight_mask] = StateStand.ST_SLIPWEIGHT

    # Create result DataFrame with columns in specified order: bvel, state_stand, state_mode
    result = pd.DataFrame(
        {
            "bvel": bvel_values,
            "state_stand": state_stand,
            "state_mode": state_mode,
        },
        index=time_aligned.index,
    )

    return result
