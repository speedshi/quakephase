### This script is used to define the phase picking object.


from obspy import UTCDateTime
from datetime import datetime
import numpy as np


def _convert_to_utcdatetime(time_value):
    '''
    Convert string or datetime to UTCDateTime 
    if input is a string or datetime. 
    Otherwise, return the original input.
    '''
    if isinstance(time_value, (str, datetime)):
        return UTCDateTime(time_value)
    return time_value


def _check_process_input1(source_id=None, receiver_id=None, phase_type=None, prob_range=None):
    '''
    Check and format the various input parameters.
    '''
    # check and process source_id
    if (source_id is not None) and (not isinstance(source_id, (str, list, tuple))):
        raise TypeError("source_id must be a string or a list of strings")
    if isinstance(source_id, str):
        source_id = [source_id]
    source_id = set(source_id)  # Convert to set for faster lookup

    # check and process receiver_id
    if (receiver_id is not None) and (not isinstance(receiver_id, (str, list, tuple))):
        raise TypeError("receiver_id must be a string or a list of strings")
    if isinstance(receiver_id, str):
        receiver_id = [receiver_id]
    receiver_id = set(receiver_id)  # Convert to set for faster lookup
    
    # check and process phase_type
    if (phase_type is not None) and (not isinstance(phase_type, (str, list, tuple))):
        raise TypeError("phase_type must be a string or a list of strings")
    if isinstance(phase_type, str):
        phase_type = [phase_type]
    phase_type = set(phase_type)  # Convert to set for faster lookup

    # check and process prob_range
    if (prob_range is not None) and (not isinstance(prob_range, (float, int, list, tuple))):
        raise TypeError("prob_range must be a number or a list of numbers")
    if isinstance(prob_range, (float, int)):
        prob_range = [prob_range, np.inf]
    elif isinstance(prob_range, (list, tuple)):
        if len(prob_range) == 1:
            # set the upper bound to infinity
            prob_range.append(np.inf)
        elif len(prob_range) == 2:
            # check if the probability range is valid
            if prob_range[0] > prob_range[1]:
                raise ValueError(f"prob_range: {prob_range} is invalid")
        else:
            raise ValueError("prob_range must be a number or a list of two numbers")
        
    return source_id, receiver_id, phase_type, prob_range


class Phase(object):
    '''
    Phase object for a paricular seismic phase of a seismic source/event.
    '''
    def __init__(self, phase_type, time, prob=None, uncert_lower=None, uncert_upper=None):
        '''
        Parameters:
            phase_type: str: type of the phase, e.g. 'P', 'S', 'Pn', 'Pg'
            time: UTCDateTime or float: time of the phase (absolute or relative)
            prob: float: probability (likelihood) of the phase, between 0 and 1
            uncert_lower: UTCDateTime or float: lower bound of the uncertainty of the phase time
            uncert_upper: UTCDateTime or float: upper bound of the uncertainty of the phase time
        
        Raises:
            TypeError: If inputs have invalid types
            ValueError: If probability is not between 0 and 1, or if time constraints are violated
        '''
        # Type validation
        if not isinstance(phase_type, str):
            raise TypeError("phase_type must be a string")
        if not isinstance(time, (UTCDateTime, datetime, int, float, str)):
            raise TypeError("time must be UTCDateTime, datetime, str, or a number")

        # convert time to UTCDateTime if it is a string or datetime
        time = _convert_to_utcdatetime(time)

        if (uncert_lower is not None):
            uncert_lower = _convert_to_utcdatetime(uncert_lower)

        if (uncert_upper is not None):
            uncert_upper = _convert_to_utcdatetime(uncert_upper)

        # Probability validation
        if prob is not None:
            if not isinstance(prob, (int, float)):
                raise TypeError("prob must be a number")
            if not 0 <= prob <= 1:
                raise ValueError("prob must be between 0 and 1")

        self.type = phase_type
        self.time = time     
        self.prob = prob     
        self.uncert_lower = uncert_lower
        self.uncert_upper = uncert_upper

        # Validate time constraints if all values are provided
        self._validate_time_constraints()

    def _validate_time_constraints(self):
        '''Validate that uncert_lower ≤ time ≤ uncert_upper when all are provided'''
        if isinstance(self.uncert_lower, UTCDateTime) and isinstance(self.time, UTCDateTime) and isinstance(self.uncert_upper, UTCDateTime):
            if self.uncert_lower > self.time:
                raise ValueError("uncert_lower must be less than or equal to phase pick time")
            if self.time > self.uncert_upper:
                raise ValueError("phase pick time must be less than or equal to uncert_upper")

    def uncertainty_range(self):
        '''
        Get time span between uncertainty bounds in seconds, 
        or None if not defined
        '''
        if (self.uncert_lower is None) or (self.uncert_upper is None):
            return None
            
        if isinstance(self.uncert_lower, UTCDateTime) and isinstance(self.uncert_upper, UTCDateTime):
            return self.uncert_upper - self.uncert_lower
        return None

    def to_dict(self):
        '''
        Convert the Phase object to a dictionary for serialization.
        
        Returns:
            dict: Dictionary representation of the Phase object
        '''
        return {
            'type': self.type,
            'time': str(self.time),
            'prob': self.prob,
            'uncert_lower': str(self.uncert_lower) if self.uncert_lower else None,
            'uncert_upper': str(self.uncert_upper) if self.uncert_upper else None
        }

    @classmethod
    def from_dict(cls, data):
        '''
        Create a Phase object from a dictionary.
        
        Parameters:
            data: dict: Dictionary containing phase data
        
        Returns:
            Phase: New Phase object
        '''
        # Check required fields
        if 'type' not in data or 'time' not in data:
            raise ValueError("Dictionary must contain 'type' and 'time' keys")

        # Convert string times to UTCDateTime if possible
        time = _convert_to_utcdatetime(data['time'])
 
        # Handle uncertainty bounds
        uncert_lower = None
        if data.get('uncert_lower'):
            uncert_lower = _convert_to_utcdatetime(data['uncert_lower'])

        uncert_upper = None
        if data.get('uncert_upper'):
            uncert_upper = _convert_to_utcdatetime(data['uncert_upper'])
        
        return cls(
            phase_type = data['type'], 
            time = time,
            prob = data.get('prob'),
            uncert_lower = uncert_lower,
            uncert_upper = uncert_upper
        )

    @classmethod
    def from_obspy_pick(cls, pick):
        """Create Phase from an ObsPy Pick object"""
        if pick.QuantityError is None:
            ilower = None
            iupper = None
        else:
            if pick.QuantityError.lower_uncertainty is None:
                ilower = None
            else:
                ilower= pick.QuantityError.lower_uncertainty
            if pick.QuantityError.upper_uncertainty is None:
                iupper = None
            else:
                iupper = pick.QuantityError.upper_uncertainty

        return cls(
            phase_type=pick.phase_hint, 
            time=pick.time,
            prob=getattr(pick, 'probability', None),
            uncert_lower=ilower, 
            uncert_upper=iupper,
        )

    @classmethod
    def create_phases(cls, phase_data_list):
        """Create multiple Phase objects (a list) from a list of dictionaries"""
        return [cls.from_dict(data) for data in phase_data_list]

    def copy(self):
        """Create a copy of this Phase object"""
        return Phase(
            phase_type=self.type,
            time=self.time,
            prob=self.prob,
            uncert_lower=self.uncert_lower,
            uncert_upper=self.uncert_upper
        )

    def __str__(self):
        return (
            f'Phase: {self.type}, time: {self.time}, prob: {self.prob}, '
            f'uncert_lower: {self.uncert_lower}, uncert_upper: {self.uncert_upper}'
        )

    def __eq__(self, other):
        '''Compare two phases for equality'''
        if not isinstance(other, Phase):
            return False
        
        return (self.type == other.type and
                str(self.time) == str(other.time) and
                self.prob == other.prob and
                str(self.uncert_lower) == str(other.uncert_lower) and
                str(self.uncert_upper) == str(other.uncert_upper))

    def __lt__(self, other):
        '''Compare two phases by time'''
        if isinstance(other, Phase):
            return self.time < other.time
        return NotImplemented

    def __repr__(self):
        '''Provide a detailed string representation for debugging'''
        return (f"Phase(phase_type='{self.type}', time={repr(self.time)}, "
                f"prob={self.prob}, uncert_lower={repr(self.uncert_lower)}, "
                f"uncert_upper={repr(self.uncert_upper)})")




class SRPhases(object):
    '''
    SRPhases object: collection of phase picks for a seismic source and receiver pair.
    i.e. a dict (phase type index) of Phase objects from the same source at a particular receiver.
    Each phase type (P, S, etc.) can only appear once in the collection, case sensitive.
    '''
    def __init__(self, phases=None, source_id=None, receiver_id=None):
        '''
        Inputs:
            phases: a list of Phase objects or None
            source_id: str or None: optional identifier for the source
            receiver_id: str or None: optional identifier for the receiver

        Raises:
            ValueError: If multiple phases of the same type are provided.
            TypeError: If a non-Phase object is provided.
        '''
        self.phases = {}  # dict of phase type: Phase object
        self.source_id = source_id
        self.receiver_id = receiver_id

        # Check for duplicate phase types during initialization
        if phases:
            if not isinstance(phases, (list, tuple)):
                raise TypeError("phases must be a list or tuple of Phase objects")

            phase_type_all = set()
            for iphase in phases:
                if not isinstance(iphase, Phase):
                    raise TypeError(f"Expected Phase object, got {type(iphase)}")
                
                if iphase.type in phase_type_all:
                    raise ValueError(f"Duplicate phase type '{iphase.type}' found. SRPhases can only contain one phase of each type.")
                
                phase_type_all.add(iphase.type)
                self.phases[iphase.type] = iphase

    def add_phase(self, phase):
        '''
        Add a Phase object to the collection.
        
        phase: Phase object to be added

        Raises:
            TypeError: If the provided object is not a Phase
            ValueError: If a phase of the same type already exists
        '''
        if not isinstance(phase, Phase):
            raise TypeError("Can only add Phase objects")
        
        # Check if phase type already exists
        if phase.type in self.phases:
            raise ValueError(f"A phase of type '{phase.type}' already exists in this collection")
        
        self.phases[phase.type] = phase

    def add_phases(self, phases):
        """Add multiple phases at once"""
        for iphase in phases:
            self.add_phase(iphase)
        return self  # For method chaining

    def remove_phase(self, phase_type=None, min_prob=None, require_both=False):
        '''
        Remove phases from the collection based on type and/or minimum probability.
        Phases of specified types and with probability less than the minimum probability are removed.
        
        Parameters:
        phase_type: str or a list/tuple of str or None
            Type(s) of phases to remove
        min_prob: float or None
            Minimum probability threshold. Phases with probability below this are removed.

        Returns:
            Directly modifies the object, no return value
        '''
        # check and process input parameters
        _, _, phase_type, _ = _check_process_input1(phase_type=phase_type)
        if (min_prob is not None) and not isinstance(min_prob, (float, int)):
            raise TypeError("min_prob must be a number")

        # remove phases directly from self.phases based on set phase_type and min_prob
        keys_to_delete = []
        for iphase in self.phases.values():
            type_match = (phase_type is not None) and (iphase.type in phase_type)
            prob_match = (min_prob is not None) and (iphase.prob is not None) and (iphase.prob < min_prob)
            
            if (require_both and type_match and prob_match) or (not require_both and (type_match or prob_match)):
                keys_to_delete.append(iphase.type)
        
        for key in keys_to_delete:
            del self.phases[key]
        
        return

    def get_phase(self, phase_type=None, prob_range=None):
        '''
        Get phase, optionally filtered by type and probability range.
        
        Parameters:
            phase_type: str or a list/tuple of str or None
                Type(s) of phases to return
            prob_range: float or a list/tuple of floats or None
                Probability range to filter by.
                If a single float, return phases with probability >= that value.
                If a list/tuple of two floats, return phases with probability in that range.
                If None, return all phases.

        Returns:
            list of Phase objects that match the criteria
        '''
        # Short-circuit if no filtering needed
        if phase_type is None and prob_range is None:
            return list(self.phases.values())

        # check and process input parameters
        _, _, phase_type, prob_range = _check_process_input1(phase_type=phase_type, 
                                                             prob_range=prob_range)  

        # filter phases based on phase_type and prob_range
        filtered_phases = [iphase for iphase in self.phases.values() if ((phase_type is None) or (iphase.type in phase_type)) and
                                                                        ((prob_range is None) or ((iphase.prob is not None) and (prob_range[0] <= iphase.prob <= prob_range[1])))
                          ]
        return filtered_phases

    def update_phase(self, phase, add_if_not_exists=False):
        '''
        Update an existing phase with new information based on phase type.
        
        Parameters:
            phase: Phase object with updated information
            add_if_not_exists: bool, optional
                If True and phase type doesn't exist, add the phase
                If False and phase type doesn't exist, raise ValueError
        
        Raises:
            TypeError: If the provided object is not a Phase
            ValueError: If the phase type doesn't exist and add_if_not_exists is False
        '''
        if not isinstance(phase, Phase):
            raise TypeError("Can only update with Phase objects")
        
        # Find the phase with the same type
        if phase.type in self.phases:
            self.phases[phase.type] = phase
            return
        
        # Phase type not found
        if add_if_not_exists:
            self.add_phase(phase)
        else:
            raise ValueError(f"No phase of type '{phase.type}' exists in this collection. Use add_phase() to add a new phase.")

    def sort_phases(self, sort_by="time", reverse=False):
        '''
        Sort phases by their time or type attribute.
        
        Parameters:
            sort_by: str, optional
                Criterion to sort by: "time" or "type"
                Default is "time"
            reverse: bool, optional
                If True, sort in descending order
                Default is False (ascending order)
        
        Returns:
            SRPhases: self reference for method chaining
        
        Raises:
            ValueError: If sort_by is not "time" or "type"
            TypeError: If sorting by time and phases have incompatible time types
        '''
        if sort_by.lower() not in ["time", "type"]:
            raise ValueError("sort_by must be either 'time' or 'type'")
        
        if sort_by.lower() == "type":
            # Sort by phase type name
            self.phases = dict(sorted(self.phases.items(), key=lambda item: item[0], reverse=reverse))
        else:  # sort_by == "time"
            self.phases = dict(sorted(self.phases.items(), key=lambda item: item[1].time, reverse=reverse))

        return self
    
    def has_phase(self, phase_type):
        '''
        Check if a phase of the specified type exists in the collection.
        
        Parameters:
            phase_type: str: Type of phase to check for
            
        Returns:
            bool: True if phase exists, False otherwise
        '''
        return phase_type in self.phases

    def get_phases_list(self):
        '''
        Get all phases as a list instead of a dictionary.
        
        Returns:
            list: List of Phase objects
        '''
        return list(self.phases.values())

    def count_phases(self):
        '''
        Count the number of phases in the collection.
        
        Returns:
            int: Number of phases
        '''
        return len(self.phases)

    def to_dict(self):
        '''
        Convert the SRPhases object to a dictionary for serialization.
        
        Returns:
            dict: Dictionary representation of the SRPhases object
        '''
        return {
            'source_id': self.source_id,
            'receiver_id': self.receiver_id,
            'phases': {phase_type: phase.to_dict() for phase_type, phase in self.phases.items()}
        }

    @classmethod
    def from_dict(cls, data):
        '''Create SRPhases object from dictionary representation'''
        phases_all = []  # list of Phase objects
        for phase_type, phase_data in data.get('phases', {}).items():
            phases_all.append(Phase.from_dict(phase_data))
            
        return cls(
            phases=phases_all,
            source_id=data.get('source_id'),
            receiver_id=data.get('receiver_id')
        )

    @classmethod
    def from_obspy_picks(cls, picks, source_id=None, receiver_id=None):
        """Create SRPhases from list of ObsPy Pick objects"""
        phases = [Phase.from_obspy_pick(ipick) for ipick in picks]
        return cls(phases=phases, source_id=source_id, receiver_id=receiver_id)

    def copy(self):
        '''Create a copy of this SRPhases object'''
        phases = [phase.copy() for phase in self.phases.values()]
        return SRPhases(
            phases=phases, 
            source_id=self.source_id, 
            receiver_id=self.receiver_id
        )

    def filter_phases(self, phase_type=None, prob_range=None, inplace=False):
        '''
        Return a new SRPhases object with phases filtered by criteria

        Parameters:
            phase_type: str or a list/tuple of str or None
                Only include these phase types
            prob_range: float or a list/tuple of floats or None
                Probability range to filter by
                when a single float, return phases with probability >= that value
                when a list/tuple of two floats, return phases with probability in that range
                when None, return all phases
            inplace: bool, optional
                If True, modify the current object in place
                If False, return a new object with the filtered phases

        Returns:
            SRPhases: A new SRPhases object containing only the phases that match the criteria
        '''
        if (prob_range is None) and (phase_type is None):
            if inplace:
                return self
            else:
                return self.copy()  # Return a simple copy if no filtering needed

        # check and process input parameters
        _, _, phase_type, prob_range = _check_process_input1(phase_type=phase_type, prob_range=prob_range)   

        # When modifying in place, we don't need to copy each phase first
        if inplace:
            filtered_phases = [
                iphase for iphase in self.phases.values()
                if ((phase_type is None) or (iphase.type in phase_type)) and
                ((prob_range is None) or ((iphase.prob is not None) and (prob_range[0] <= iphase.prob <= prob_range[1])))
            ]
            self.phases = {phase.type: phase for phase in filtered_phases}
            return self
        else:
            # Only copy when creating a new object
            filtered_phases = [
                iphase.copy() for iphase in self.phases.values()
                if ((phase_type is None) or (iphase.type in phase_type)) and
                ((prob_range is None) or ((iphase.prob is not None) and (prob_range[0] <= iphase.prob <= prob_range[1])))
            ]
            return SRPhases(phases=filtered_phases, source_id=self.source_id, receiver_id=self.receiver_id)

    def validate_phases(self):
        '''
        Validate all phases in the collection. 
        Returns list of issues or empty list if valid.
        '''
        issues = []
        # Validate time relationships (e.g., S must come after P if both exist)
        if 'P' in self.phases and 'S' in self.phases:
            if self.phases['S'].time < self.phases['P'].time:
                issues.append("S phase must come after P phase")
        
        # Add more validation rules here
        return issues

    def __str__(self):
        phases_str = '\n  '.join(str(phase) for phase in self.phases.values())
        source_info = f'Source ID: {self.source_id}, ' if self.source_id else ''
        return f'SRPhases ({source_info}{len(self.phases)} phases):\n  {phases_str}'

    def __eq__(self, other):
        '''
        Compare two SRPhases objects for equality.
        
        Returns:
            bool: True if objects are equal, False otherwise
        '''
        # Quick checks first to fail fast
        if not isinstance(other, SRPhases):
            return False
        if (self.source_id != other.source_id) or (self.receiver_id != other.receiver_id):
            return False
        if len(self.phases) != len(other.phases):
            return False
        
        return (self.phases.keys() == other.phases.keys() and
                all(self.phases[k] == other.phases[k] for k in self.phases))  # Use Phase.__eq__ here




class Picks(object):
    '''
    A collection of phase picks. 
    The picks may come from different sources and different receivers.
    '''
    def __init__(self, SRPhase_list=[]):
        '''
        SRPhase_list: a list of SRPhase_list objects or a single SRPhase_list object
        '''
        # Validate input and initialize the list of SRPhases objects
        if isinstance(SRPhase_list, (list, tuple)):
            for iphase in SRPhase_list:
                if not isinstance(iphase, SRPhases):
                    raise TypeError("SRPhase_list must contain SRPhases objects")
            self.picks = SRPhase_list
        elif isinstance(SRPhase_list, SRPhases):
            self.picks = [SRPhase_list]
        else:
            raise TypeError("SRPhase_list must be a list or tuple of SRPhases objects or a single SRPhases object")
        
    def add_picks(self, SRPhase_list):
        '''
        Add a list/single of SRPhases object to the collection.
        
        Parameters:
            SRPhase_list: list of SRPhases objects
        '''
        if isinstance(SRPhase_list, (list, tuple)):
            self.picks.extend(SRPhase_list)
        elif isinstance(SRPhase_list, SRPhases):
            self.picks.append(SRPhase_list)
        else:
            raise TypeError("SRPhase_list must be a list or tuple of SRPhases objects or a single SRPhases object")

    def get_picks(self, source_id=None, receiver_id=None, phase_type=None, prob_range=None):
        '''
        Filtered picks by source_id, receiver_id, phase_type, and probability range.
        
        Parameters:
            source_id: str or a list/tuple of str or None
                Source ID to filter by
            receiver_id: str or a list/tuple of str or None
                Receiver ID to filter by
            phase_type: str or a list/tuple of str or None
                Phase type to filter by
            prob_range: float or a list/tuple of floats or None
                Probability range or minimum probability threshold to filter by
        
        Returns:
            Returing a new Picks object with the filtered list of SRPhases objects
            without modifying the original.
        '''
        # Short-circuit if no filtering needed
        if (source_id is None) and (receiver_id is None) and (phase_type is None) and (prob_range is None):
            return Picks(SRPhase_list=self.picks.copy())
        
        # check and process input parameters
        source_id, receiver_id, phase_type, prob_range = _check_process_input1(source_id=source_id, 
                                                                               receiver_id=receiver_id, 
                                                                               phase_type=phase_type,
                                                                               prob_range=prob_range)

        filtered_picks = []
        for ipick in self.picks:
            # Check source_id and receiver_id first (faster checks)
            if (source_id is not None) and (ipick.source_id not in source_id):
                continue
            if (receiver_id is not None) and (ipick.receiver_id not in receiver_id):
                continue
            
            # If phase_type filtering is requested, check for matching phases
            if (phase_type is None) and (prob_range is None):
                filtered_picks.append(ipick.copy())  # Just copy the original
            else:
                # Get all phases of the specified types with probability in range
                matching_pick = ipick.filter_phases(phase_type=phase_type, prob_range=prob_range, inplace=False)
                if matching_pick.count_phases() > 0:  # Only add if it has phases
                    filtered_picks.append(matching_pick)
        
        # filtered_picks = [
        #     pick for pick in [
        #         ipick.filter_phases(phase_type=phase_type, prob_range=prob_range) 
        #         for ipick in self.picks
        #         if ((source_id is None) or (ipick.source_id in source_id)) and
        #         ((receiver_id is None) or (ipick.receiver_id in receiver_id))
        #     ] if pick.count_phases() > 0
        # ]

        return Picks(SRPhase_list=filtered_picks)

    def remove_picks(self, source_id=None, receiver_id=None):
        '''
        Remove phase picks filtered by source and/or receiver ID.
        Inputs:
            source_id: str or a list/tuple of str or None: source ID to filter by
            receiver_id: str or a list/tuple of str or None: receiver ID to filter by

        Returns:
            None
        '''
        if (source_id is None) and (receiver_id is None):
            # no removal needed
            return
        

        # remove picks
        self.picks = [ipick for ipick in self.picks if (source_id is None or ipick.source_id not in source_id) and (receiver_id is None or ipick.receiver_id not in receiver_id)]
        return

    
