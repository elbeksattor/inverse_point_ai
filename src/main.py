#!/usr/bin/env python3
"""
IP AI v3 - Main Entry Point

Inverse Point AI Analytics System for real-time person detection,
tracking, and analytics using NVIDIA DeepStream SDK.
"""

import sys
import os
import argparse
import signal
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.utils.logger import setup_logger, get_logger
from src.utils.config import load_config, PipelineConfig
from src.pipeline.camera_pipeline import CameraPipeline, FrameMetadata
from src.database.person_database import PersonDatabase


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IP AI v3 - Real-time Person Analytics System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with USB camera (default)
  python src/main.py --camera /dev/video0

  # Run with custom resolution
  python src/main.py --camera /dev/video0 --width 1920 --height 1080

  # Run without display (headless)
  python src/main.py --camera /dev/video0 --no-display

  # Run with config file
  python src/main.py --config configs/pipeline_config.yaml
"""
    )

    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--camera", "-c",
        default="/dev/video0",
        help="Camera device path (default: /dev/video0)"
    )
    input_group.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file"
    )

    # Video options
    video_group = parser.add_argument_group("Video Options")
    video_group.add_argument(
        "--width", "-W",
        type=int, default=1280,
        help="Frame width (default: 1280)"
    )
    video_group.add_argument(
        "--height", "-H",
        type=int, default=720,
        help="Frame height (default: 720)"
    )
    video_group.add_argument(
        "--fps", "-f",
        type=int, default=30,
        help="Target framerate (default: 30)"
    )

    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--pgie-config",
        default="configs/pgie_config.txt",
        help="Primary GIE config file"
    )
    model_group.add_argument(
        "--tracker-config",
        default="configs/tracker_config.txt",
        help="Tracker config file (use 'none' to disable)"
    )
    model_group.add_argument(
        "--sgie-config",
        default=None,
        help="Demographics SGIE config file (default: disabled)"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display output (headless mode)"
    )
    output_group.add_argument(
        "--output", "-o",
        default=None,
        help="Output video file path"
    )

    # RE-ID Database options
    db_group = parser.add_argument_group("RE-ID Database Options")
    db_group.add_argument(
        "--no-database",
        action="store_true",
        help="Disable persistent RE-ID database"
    )
    db_group.add_argument(
        "--db-path",
        default="output/person_database.db",
        help="Path to SQLite database file"
    )
    db_group.add_argument(
        "--reid-threshold",
        type=float,
        default=0.50,
        help="Cosine similarity threshold for RE-ID matching (default: 0.50, adaptive)"
    )

    # Logging options
    log_group = parser.add_argument_group("Logging Options")
    log_group.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)"
    )
    log_group.add_argument(
        "--log-file",
        default=None,
        help="Log file path"
    )

    return parser.parse_args()


def bbox_overlap(box1, box2):
    """Check if box1 is mostly inside box2 (face inside person)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area1 = w1 * h1

    # Return ratio of face that is inside person
    return intersection / area1 if area1 > 0 else 0.0


def create_frame_callback(logger, person_database=None):
    """Create frame callback for logging detections."""
    frame_stats = {
        'total_persons': 0,
        'total_frames': 0,
        'unique_person_ids': set()  # Track unique Person IDs
    }

    def on_frame(frame_data: FrameMetadata):
        """Process frame metadata."""
        frame_stats['total_frames'] += 1

        # Count persons in this frame
        persons = [d for d in frame_data.detections if d.class_name == 'person']
        faces = [d for d in frame_data.detections if d.class_name == 'face']
        frame_stats['total_persons'] += len(persons)

        # Track unique Person IDs
        for p in persons:
            if p.person_id is not None:
                frame_stats['unique_person_ids'].add(p.person_id)

        # Associate faces with persons by bbox overlap and update demographics
        if person_database and faces and persons:
            for face in faces:
                if face.age is None and face.gender is None:
                    continue  # No demographics for this face

                # Find the person with highest overlap
                best_person = None
                best_overlap = 0.5  # Minimum 50% overlap required

                for person in persons:
                    if person.person_id is None:
                        continue
                    overlap = bbox_overlap(face.bbox, person.bbox)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_person = person

                # Update person demographics in database
                if best_person is not None:
                    try:
                        person_database.update_demographics(
                            person_id=best_person.person_id,
                            age_group=face.age_group,
                            gender=face.gender
                        )
                    except Exception as e:
                        logger.debug("demographics_update_failed", error=str(e))

        # Log every 30 frames (~1 second at 30fps)
        if frame_data.frame_num % 30 == 0:
            # Get Person IDs (persistent) and Track IDs (session)
            person_ids = [p.person_id for p in persons if p.person_id is not None]
            track_ids = [p.tracker_id for p in persons if p.tracker_id > 0]

            person_str = ",".join(str(pid) for pid in person_ids) if person_ids else "none"
            track_str = ",".join(str(tid) for tid in track_ids) if track_ids else "none"

            # Get demographics info from faces
            demo_str = ""
            for face in faces:
                if face.age is not None:
                    demo_str = f" | Demo: {face.gender[0] if face.gender else '?'}{face.age}"
                    break

            print(
                f"[Frame {frame_data.frame_num:5d}] "
                f"Persons: {len(persons):2d} | "
                f"Faces: {len(faces):2d} | "
                f"IDs: [P{person_str}] | "
                f"FPS: {frame_data.fps:5.1f} | "
                f"Unique: {len(frame_stats['unique_person_ids'])}"
                f"{demo_str}"
            )

            # Log detection details if any
            for det in persons:
                logger.debug(
                    "detection",
                    class_name=det.class_name,
                    person_id=det.person_id,
                    tracker_id=det.tracker_id,
                    confidence=f"{det.confidence:.2f}",
                    bbox=det.bbox
                )

    return on_frame, frame_stats


def main():
    """Main entry point."""
    args = parse_args()

    # Set up logging
    log_levels = {
        "debug": 10,
        "info": 20,
        "warning": 30,
        "error": 40
    }
    logger = setup_logger(
        level=log_levels[args.log_level],
        log_file=args.log_file
    )

    logger.info(
        "ip_ai_v3_starting",
        version="3.0.0",
        camera=args.camera,
        resolution=f"{args.width}x{args.height}",
        fps=args.fps
    )

    # Load config file if provided
    config = None
    if args.config:
        config = load_config(args.config)
        logger.info("config_loaded", path=args.config)

    # Handle tracker config
    tracker_config = args.tracker_config
    if tracker_config.lower() == 'none':
        tracker_config = None
    elif not Path(tracker_config).exists():
        logger.warning("tracker_config_not_found", path=tracker_config)
        tracker_config = None

    # Verify camera device
    if not Path(args.camera).exists():
        logger.error("camera_not_found", device=args.camera)
        print(f"\nError: Camera device not found: {args.camera}")
        print("\nAvailable video devices:")
        os.system("ls -la /dev/video* 2>/dev/null || echo '  No video devices found'")
        sys.exit(1)

    # Verify model files
    pgie_config = Path(args.pgie_config)
    if not pgie_config.exists():
        logger.error("pgie_config_not_found", path=str(pgie_config))
        sys.exit(1)

    # Initialize RE-ID database (unless disabled)
    person_database = None
    if not args.no_database:
        try:
            person_database = PersonDatabase(
                db_path=args.db_path,
                index_path=args.db_path.replace('.db', '.index'),
                embedding_dim=256  # ResNet50 Market1501
            )
            logger.info(
                "person_database_enabled",
                db_path=args.db_path,
                reid_threshold=args.reid_threshold
            )
        except Exception as e:
            logger.warning("person_database_init_failed", error=str(e))
            person_database = None

    # Create frame callback
    on_frame, frame_stats = create_frame_callback(logger, person_database)

    # Handle SGIE config
    sgie_config = args.sgie_config
    if sgie_config and sgie_config.lower() == 'none':
        sgie_config = None
    elif sgie_config and not Path(sgie_config).exists():
        logger.warning("sgie_config_not_found", path=sgie_config)
        sgie_config = None

    # Create pipeline
    pipeline = CameraPipeline(
        camera_device=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        pgie_config=str(pgie_config),
        tracker_config=tracker_config,
        sgie_config=sgie_config,
        output_file=args.output,
        display=not args.no_display,
        on_frame_callback=on_frame,
        person_database=person_database,
        reid_threshold=args.reid_threshold
    )

    # Set up signal handlers
    shutdown_requested = [False]  # Use list to allow modification in nested function

    def signal_handler(sig, frame):
        if shutdown_requested[0]:
            return  # Already shutting down
        shutdown_requested[0] = True
        print("\n\nReceived interrupt signal, stopping...")
        pipeline.stop()
        # Don't call sys.exit() - let the finally block handle cleanup

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run pipeline
    print("\n" + "=" * 60)
    print("IP AI v3 - Person Detection Pipeline")
    print("=" * 60)
    print(f"Camera:     {args.camera}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
    print(f"Display:    {'Enabled' if not args.no_display else 'Disabled'}")
    print(f"Tracker:    {'Enabled' if tracker_config else 'Disabled'}")
    print(f"RE-ID DB:   {'Enabled' if person_database else 'Disabled'}")
    print(f"Demographics: {'Enabled' if sgie_config else 'Disabled'}")
    print("=" * 60)
    print("Press Ctrl+C to stop\n")

    try:
        pipeline.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error("pipeline_error", error=str(e))
        raise
    finally:
        pipeline.stop()

        # Get database statistics and save
        db_unique_count = 0
        if person_database:
            try:
                db_unique_count = person_database.get_unique_persons_count()
                # Save database on exit (CRITICAL for persistence!)
                print("Saving RE-ID database...")
                person_database.save_index()
                person_database.close()
                print(f"Database saved: {db_unique_count} unique persons")
                logger.info("person_database_closed", unique_persons=db_unique_count)
            except Exception as e:
                logger.warning("database_close_error", error=str(e))
                print(f"Error saving database: {e}")

        # Print final statistics
        print("\n" + "=" * 60)
        print("Session Statistics")
        print("=" * 60)
        print(f"Total Frames:   {frame_stats['total_frames']}")
        print(f"Total Persons:  {frame_stats['total_persons']}")
        print(f"Unique (session): {len(frame_stats['unique_person_ids'])}")
        print(f"Unique (all-time): {db_unique_count}")
        print(f"Avg FPS:        {pipeline.get_fps():.1f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
