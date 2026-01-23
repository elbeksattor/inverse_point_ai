#!/usr/bin/env python3
"""
Professional Tracking Analysis System
Analyzes ReID tracking data to diagnose ID switching issues

This tool provides:
1. Tracker-to-Person mapping timeline analysis
2. Embedding similarity clustering
3. ID switch event detection and categorization
4. Visual timeline generation
5. Root cause diagnosis
"""

import sys
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class TrackingEvent:
    """Single tracking event"""
    frame: int
    tracker_id: int
    person_id: int
    event_type: str  # 'created', 'reidentified', 'corrected', 'switch'
    confidence: float = 0.0
    similarity: float = 0.0
    bbox: Tuple[float, float, float, float] = None
    embedding: np.ndarray = None
    details: Dict = field(default_factory=dict)


@dataclass
class PersonProfile:
    """Profile for a unique person"""
    person_id: int
    first_frame: int
    last_frame: int
    total_appearances: int
    tracker_ids: Set[int] = field(default_factory=set)
    embeddings: List[np.ndarray] = field(default_factory=list)
    avg_embedding: np.ndarray = None
    bbox_history: List[Tuple[int, Tuple]] = field(default_factory=list)  # (frame, bbox)


@dataclass
class TrackerProfile:
    """Profile for a tracker ID"""
    tracker_id: int
    first_frame: int
    last_frame: int
    person_ids: List[Tuple[int, int, int]] = field(default_factory=list)  # (start_frame, end_frame, person_id)
    id_switches: int = 0


class TrackingAnalyzer:
    """
    Comprehensive tracking analysis system
    """

    def __init__(self, log_path: str, db_path: str = None):
        self.log_path = Path(log_path)
        self.db_path = Path(db_path) if db_path else None

        self.events: List[TrackingEvent] = []
        self.persons: Dict[int, PersonProfile] = {}
        self.trackers: Dict[int, TrackerProfile] = {}

        # Analysis results
        self.id_switches: List[Dict] = []
        self.embedding_clusters: List[Dict] = []
        self.trajectory_overlaps: List[Dict] = []
        self.diagnosis: Dict = {}

    def parse_log(self) -> None:
        """Parse log file and extract tracking events"""
        print(f"Parsing log file: {self.log_path}")

        with open(self.log_path, 'r') as f:
            for line in f:
                try:
                    # Parse JSON log entry
                    if line.strip().startswith('{'):
                        entry = eval(line.strip())  # Using eval for numpy types
                        self._process_log_entry(entry)
                except Exception as e:
                    continue

        print(f"Parsed {len(self.events)} tracking events")

    def _process_log_entry(self, entry: Dict) -> None:
        """Process a single log entry"""
        event_type = entry.get('event', '')

        if event_type == 'new_person_detected':
            self.events.append(TrackingEvent(
                frame=entry.get('frame', 0),
                tracker_id=entry.get('tracker_id', 0),
                person_id=entry.get('person_id', 0),
                event_type='created',
                similarity=entry.get('similarity', 0.0)
            ))

        elif event_type == 'person_reidentified':
            self.events.append(TrackingEvent(
                frame=entry.get('frame', 0),
                tracker_id=entry.get('tracker_id', 0),
                person_id=entry.get('person_id', 0),
                event_type='reidentified',
                similarity=entry.get('similarity', 0.0)
            ))

        elif event_type in ['id_switch_corrected', 'id_switch_corrected_drift', 'id_switch_corrected_mismatch']:
            self.events.append(TrackingEvent(
                frame=entry.get('frame', 0),
                tracker_id=entry.get('tracker_id', 0),
                person_id=entry.get('new_person_id', 0),
                event_type='corrected',
                similarity=entry.get('db_match_similarity', 0.0),
                details={
                    'old_person_id': entry.get('old_person_id'),
                    'embedding_similarity': entry.get('embedding_similarity'),
                    'correction_type': event_type
                }
            ))

        elif event_type == 'id_switch_prevented_new_person':
            self.events.append(TrackingEvent(
                frame=entry.get('frame', 0),
                tracker_id=entry.get('tracker_id', 0),
                person_id=entry.get('corrected_to_person_id', 0),
                event_type='prevented',
                similarity=entry.get('sim_to_prev', 0.0),
                details={
                    'would_be_new_person_id': entry.get('would_be_new_person_id'),
                    'previous_person_id': entry.get('previous_person_id'),
                    'best_db_similarity': entry.get('best_db_similarity'),
                    'reason': entry.get('reason')
                }
            ))

        elif event_type == 'reid_embedding_extracted':
            # Track embedding extraction for analysis
            pass

    def build_profiles(self) -> None:
        """Build person and tracker profiles from events"""
        print("Building person and tracker profiles...")

        # Sort events by frame
        self.events.sort(key=lambda e: (e.frame, e.tracker_id))

        # Build tracker timeline
        tracker_timeline: Dict[int, List[Tuple[int, int]]] = defaultdict(list)  # tracker_id -> [(frame, person_id)]

        for event in self.events:
            tracker_timeline[event.tracker_id].append((event.frame, event.person_id))

            # Update person profile
            if event.person_id not in self.persons:
                self.persons[event.person_id] = PersonProfile(
                    person_id=event.person_id,
                    first_frame=event.frame,
                    last_frame=event.frame,
                    total_appearances=1
                )
            else:
                p = self.persons[event.person_id]
                p.last_frame = max(p.last_frame, event.frame)
                p.total_appearances += 1

            self.persons[event.person_id].tracker_ids.add(event.tracker_id)

        # Build tracker profiles with ID switch detection
        for tracker_id, timeline in tracker_timeline.items():
            if not timeline:
                continue

            profile = TrackerProfile(
                tracker_id=tracker_id,
                first_frame=timeline[0][0],
                last_frame=timeline[-1][0]
            )

            # Detect person ID changes within this tracker
            current_person = timeline[0][1]
            segment_start = timeline[0][0]

            for frame, person_id in timeline:
                if person_id != current_person:
                    # ID switch detected
                    profile.person_ids.append((segment_start, frame - 1, current_person))
                    profile.id_switches += 1
                    current_person = person_id
                    segment_start = frame

            # Add final segment
            profile.person_ids.append((segment_start, timeline[-1][0], current_person))

            self.trackers[tracker_id] = profile

        print(f"Built profiles for {len(self.persons)} persons and {len(self.trackers)} trackers")

    def analyze_id_switches(self) -> None:
        """Analyze all ID switch events"""
        print("\nAnalyzing ID switches...")

        for tracker_id, profile in self.trackers.items():
            if profile.id_switches > 0:
                for i in range(len(profile.person_ids) - 1):
                    start1, end1, person1 = profile.person_ids[i]
                    start2, end2, person2 = profile.person_ids[i + 1]

                    self.id_switches.append({
                        'tracker_id': tracker_id,
                        'frame': start2,
                        'from_person': person1,
                        'to_person': person2,
                        'from_duration': end1 - start1 + 1,
                        'segment_before': (start1, end1),
                        'segment_after': (start2, end2)
                    })

        # Sort by frame
        self.id_switches.sort(key=lambda x: x['frame'])

        print(f"Found {len(self.id_switches)} ID switch events")

    def analyze_person_fragmentation(self) -> Dict:
        """Analyze how persons are fragmented across multiple IDs"""
        print("\nAnalyzing person fragmentation...")

        fragmentation = {}

        for person_id, profile in self.persons.items():
            # Calculate coverage gaps
            frame_range = profile.last_frame - profile.first_frame + 1
            coverage = profile.total_appearances / max(frame_range, 1)

            fragmentation[person_id] = {
                'person_id': person_id,
                'first_frame': profile.first_frame,
                'last_frame': profile.last_frame,
                'total_appearances': profile.total_appearances,
                'frame_range': frame_range,
                'coverage': coverage,
                'tracker_ids': list(profile.tracker_ids),
                'num_trackers': len(profile.tracker_ids)
            }

        return fragmentation

    def find_potential_duplicates(self) -> List[Dict]:
        """Find persons that might be the same real person"""
        print("\nFinding potential duplicate persons...")

        duplicates = []
        person_ids = list(self.persons.keys())

        for i, p1_id in enumerate(person_ids):
            p1 = self.persons[p1_id]

            for p2_id in person_ids[i+1:]:
                p2 = self.persons[p2_id]

                # Check for temporal overlap or near-sequential appearance
                overlap = self._check_temporal_relationship(p1, p2)

                if overlap['relationship'] != 'no_relation':
                    duplicates.append({
                        'person_1': p1_id,
                        'person_2': p2_id,
                        'relationship': overlap['relationship'],
                        'gap_frames': overlap.get('gap', 0),
                        'shared_trackers': list(p1.tracker_ids & p2.tracker_ids)
                    })

        return duplicates

    def _check_temporal_relationship(self, p1: PersonProfile, p2: PersonProfile) -> Dict:
        """Check temporal relationship between two persons"""
        # Check if they share any tracker IDs
        shared_trackers = p1.tracker_ids & p2.tracker_ids

        if shared_trackers:
            # They appeared on the same tracker - potential ID switch
            return {
                'relationship': 'same_tracker',
                'shared_trackers': list(shared_trackers)
            }

        # Check temporal proximity
        if p1.last_frame < p2.first_frame:
            gap = p2.first_frame - p1.last_frame
            if gap < 30:  # Within 1 second at 30fps
                return {'relationship': 'sequential', 'gap': gap}
        elif p2.last_frame < p1.first_frame:
            gap = p1.first_frame - p2.last_frame
            if gap < 30:
                return {'relationship': 'sequential', 'gap': gap}

        # Check overlap
        overlap_start = max(p1.first_frame, p2.first_frame)
        overlap_end = min(p1.last_frame, p2.last_frame)

        if overlap_start <= overlap_end:
            return {
                'relationship': 'overlapping',
                'overlap_frames': overlap_end - overlap_start + 1
            }

        return {'relationship': 'no_relation'}

    def diagnose_issues(self) -> Dict:
        """Generate comprehensive diagnosis of tracking issues"""
        print("\n" + "="*60)
        print("TRACKING SYSTEM DIAGNOSIS")
        print("="*60)

        diagnosis = {
            'summary': {},
            'issues': [],
            'root_causes': [],
            'recommendations': []
        }

        # Summary statistics
        total_persons = len(self.persons)
        total_trackers = len(self.trackers)
        total_switches = sum(t.id_switches for t in self.trackers.values())

        diagnosis['summary'] = {
            'total_persons_detected': total_persons,
            'total_trackers_used': total_trackers,
            'total_id_switches': total_switches,
            'avg_switches_per_tracker': total_switches / max(total_trackers, 1)
        }

        print(f"\nSummary:")
        print(f"  - Total persons detected: {total_persons}")
        print(f"  - Total trackers used: {total_trackers}")
        print(f"  - Total ID switches: {total_switches}")

        # Analyze fragmentation
        fragmentation = self.analyze_person_fragmentation()

        # Find highly fragmented trackers (likely problematic)
        problematic_trackers = []
        for tid, profile in self.trackers.items():
            if len(profile.person_ids) > 2:
                problematic_trackers.append({
                    'tracker_id': tid,
                    'num_persons': len(profile.person_ids),
                    'persons': [p[2] for p in profile.person_ids],
                    'frame_range': f"{profile.first_frame}-{profile.last_frame}"
                })

        if problematic_trackers:
            diagnosis['issues'].append({
                'type': 'TRACKER_FRAGMENTATION',
                'severity': 'HIGH',
                'description': f'{len(problematic_trackers)} trackers assigned to multiple persons',
                'details': problematic_trackers
            })

            print(f"\n⚠️  ISSUE: Tracker Fragmentation")
            print(f"   {len(problematic_trackers)} trackers were assigned to multiple persons:")
            for pt in problematic_trackers:
                print(f"   - Tracker {pt['tracker_id']}: {pt['num_persons']} persons {pt['persons']} "
                      f"(frames {pt['frame_range']})")

        # Find potential duplicates
        duplicates = self.find_potential_duplicates()
        same_tracker_dups = [d for d in duplicates if d['relationship'] == 'same_tracker']

        if same_tracker_dups:
            diagnosis['issues'].append({
                'type': 'DUPLICATE_PERSONS',
                'severity': 'HIGH',
                'description': f'{len(same_tracker_dups)} potential duplicate person entries',
                'details': same_tracker_dups
            })

            print(f"\n⚠️  ISSUE: Potential Duplicate Persons")
            print(f"   {len(same_tracker_dups)} person pairs share the same tracker:")
            for dup in same_tracker_dups:
                print(f"   - Person {dup['person_1']} & Person {dup['person_2']} "
                      f"(shared trackers: {dup['shared_trackers']})")

        # Analyze ID switch patterns
        if self.id_switches:
            # Group switches by tracker
            switches_by_tracker = defaultdict(list)
            for switch in self.id_switches:
                switches_by_tracker[switch['tracker_id']].append(switch)

            print(f"\n⚠️  ISSUE: ID Switches by Tracker")
            for tid, switches in sorted(switches_by_tracker.items()):
                print(f"   Tracker {tid}: {len(switches)} switches")
                for s in switches:
                    print(f"     Frame {s['frame']}: Person {s['from_person']} → Person {s['to_person']}")

        # Root cause analysis
        print("\n" + "-"*60)
        print("ROOT CAUSE ANALYSIS")
        print("-"*60)

        # Check for trajectory-based issues
        if problematic_trackers:
            # Find trackers with rapid person changes
            rapid_changers = [t for t in problematic_trackers
                           if len(t['persons']) >= 3]
            if rapid_changers:
                diagnosis['root_causes'].append({
                    'cause': 'TRAJECTORY_BASED_TRACKING',
                    'confidence': 'HIGH',
                    'description': 'NvDCF tracker is reassigning tracker IDs based on trajectory '
                                 'proximity rather than visual appearance',
                    'evidence': f'{len(rapid_changers)} trackers with 3+ person assignments',
                    'affected_trackers': [t['tracker_id'] for t in rapid_changers]
                })
                print("\n🔍 ROOT CAUSE 1: Trajectory-Based Tracking")
                print("   The NvDCF tracker prioritizes spatial proximity over visual similarity.")
                print("   When people cross paths or stand close, tracker IDs swap between them.")
                print(f"   Evidence: {len(rapid_changers)} trackers with 3+ person assignments")

        # Check for ReID threshold issues
        prevented_events = [e for e in self.events if e.event_type == 'prevented']
        if prevented_events:
            low_sim_prevents = [e for e in prevented_events
                              if e.similarity < 0.7 and e.details.get('reason') == 'still_matches_previous']
            if low_sim_prevents:
                diagnosis['root_causes'].append({
                    'cause': 'REID_THRESHOLD_TOO_LOW',
                    'confidence': 'MEDIUM',
                    'description': 'ReID similarity threshold may be allowing false matches',
                    'evidence': f'{len(low_sim_prevents)} preventions with similarity < 0.7'
                })
                print("\n🔍 ROOT CAUSE 2: ReID Threshold Configuration")
                print("   Some ID corrections occurred with relatively low similarity scores.")
                print(f"   Evidence: {len(low_sim_prevents)} corrections with similarity < 0.7")

        # Check for occlusion patterns
        short_lived_persons = [p for p in fragmentation.values()
                             if p['total_appearances'] < 20 and p['frame_range'] > 50]
        if short_lived_persons:
            diagnosis['root_causes'].append({
                'cause': 'OCCLUSION_HANDLING',
                'confidence': 'MEDIUM',
                'description': 'Persons are being lost and re-created after occlusions',
                'evidence': f'{len(short_lived_persons)} persons with low coverage over long frame ranges'
            })
            print("\n🔍 ROOT CAUSE 3: Occlusion Handling")
            print("   Some persons have low detection coverage over their frame range,")
            print("   suggesting they are lost and re-created after occlusions.")
            print(f"   Evidence: {len(short_lived_persons)} persons with sparse detections")

        # Generate recommendations
        print("\n" + "-"*60)
        print("RECOMMENDATIONS")
        print("-"*60)

        recommendations = []

        if any(rc['cause'] == 'TRAJECTORY_BASED_TRACKING' for rc in diagnosis['root_causes']):
            recommendations.append({
                'priority': 1,
                'action': 'INCREASE_REID_WEIGHT',
                'description': 'Increase ReID weight in tracker configuration',
                'config_changes': [
                    'matchingScoreWeight4ReidSimilarity: 0.85 (from 0.70)',
                    'matchingScoreWeight4TrackletSimilarity: 0.30 (from 0.50)',
                    'minMatchingScore4ReidSimilarity: 0.20 (from 0.15)'
                ]
            })
            print("\n1️⃣  INCREASE ReID Weight in Tracker")
            print("    Modify nvdcf_tracker_config.yml:")
            print("    - matchingScoreWeight4ReidSimilarity: 0.85")
            print("    - matchingScoreWeight4TrackletSimilarity: 0.30")

        recommendations.append({
            'priority': 2,
            'action': 'IMPLEMENT_GLOBAL_REID',
            'description': 'Add global ReID database lookup for all detections',
            'implementation': 'Before assigning tracker result, verify against global person database'
        })
        print("\n2️⃣  Implement Global ReID Verification")
        print("    For every detection, verify the person ID against the global database,")
        print("    not just the tracker history.")

        recommendations.append({
            'priority': 3,
            'action': 'ADD_EMBEDDING_AVERAGING',
            'description': 'Maintain running average of embeddings per person',
            'implementation': 'Use exponential moving average of embeddings for more stable matching'
        })
        print("\n3️⃣  Use Embedding Averaging")
        print("    Maintain a running average of embeddings per person for more stable matching.")

        diagnosis['recommendations'] = recommendations
        self.diagnosis = diagnosis

        return diagnosis

    def generate_timeline_report(self) -> str:
        """Generate a visual timeline report"""
        report = []
        report.append("\n" + "="*80)
        report.append("TRACKER-PERSON TIMELINE")
        report.append("="*80)

        # Find frame range
        all_frames = [e.frame for e in self.events]
        if not all_frames:
            return "No events to report"

        min_frame = min(all_frames)
        max_frame = max(all_frames)

        # Create timeline for each tracker
        for tracker_id in sorted(self.trackers.keys()):
            profile = self.trackers[tracker_id]

            report.append(f"\nTracker {tracker_id}:")

            for start, end, person_id in profile.person_ids:
                duration = end - start + 1
                bar = "█" * min(duration // 10, 30)
                report.append(f"  Frame {start:4d}-{end:4d}: Person {person_id:2d} {bar}")

            if profile.id_switches > 0:
                report.append(f"  ⚠️  {profile.id_switches} ID switch(es)")

        return "\n".join(report)

    def generate_person_report(self) -> str:
        """Generate person-centric report"""
        report = []
        report.append("\n" + "="*80)
        report.append("PERSON TRACKING REPORT")
        report.append("="*80)

        fragmentation = self.analyze_person_fragmentation()

        for person_id in sorted(self.persons.keys()):
            frag = fragmentation[person_id]
            profile = self.persons[person_id]

            report.append(f"\nPerson {person_id}:")
            report.append(f"  Frames: {frag['first_frame']} - {frag['last_frame']} "
                         f"(range: {frag['frame_range']})")
            report.append(f"  Appearances: {frag['total_appearances']} "
                         f"(coverage: {frag['coverage']*100:.1f}%)")
            report.append(f"  Trackers: {frag['tracker_ids']}")

            if frag['num_trackers'] > 1:
                report.append(f"  ⚠️  Multiple trackers - possible fragmentation")

        return "\n".join(report)

    def save_analysis(self, output_path: str) -> None:
        """Save complete analysis to JSON file"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'log_file': str(self.log_path),
            'diagnosis': self.diagnosis,
            'id_switches': self.id_switches,
            'person_fragmentation': self.analyze_person_fragmentation(),
            'potential_duplicates': self.find_potential_duplicates(),
            'tracker_profiles': {
                tid: {
                    'first_frame': p.first_frame,
                    'last_frame': p.last_frame,
                    'person_segments': p.person_ids,
                    'id_switches': p.id_switches
                }
                for tid, p in self.trackers.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        print(f"\nAnalysis saved to: {output_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze tracking system performance')
    parser.add_argument('--log', type=str, required=True, help='Path to log file')
    parser.add_argument('--db', type=str, help='Path to person database')
    parser.add_argument('--output', type=str, help='Output path for analysis JSON')

    args = parser.parse_args()

    analyzer = TrackingAnalyzer(args.log, args.db)
    analyzer.parse_log()
    analyzer.build_profiles()
    analyzer.analyze_id_switches()

    # Generate reports
    print(analyzer.generate_timeline_report())
    print(analyzer.generate_person_report())

    # Diagnose issues
    diagnosis = analyzer.diagnose_issues()

    # Save analysis
    if args.output:
        analyzer.save_analysis(args.output)
    else:
        # Default output path
        output_path = Path(args.log).parent / 'tracking_analysis.json'
        analyzer.save_analysis(str(output_path))


if __name__ == '__main__':
    main()
