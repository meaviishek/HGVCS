#!/usr/bin/env python3
"""
HGVCS - Hand Gesture & Voice Control System
Main Application Entry Point

A next-generation human-computer interaction platform that enables
users to control their computing environment through hand gestures
and natural voice commands.

Author: HGVCS Team
Version: 1.0.0
License: MIT
"""

import sys
import os
import argparse
import signal
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Check Python version
if sys.version_info < (3, 8):
    print("Error: Python 3.8 or higher is required")
    sys.exit(1)

def setup_logging(config):
    """Setup logging configuration."""
    from colorlog import ColoredFormatter
    
    log_level = getattr(logging, config.get('level', 'INFO'))
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logger
    logger = logging.getLogger('hgvcs')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    if config.get('console_output', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        colored_formatter = ColoredFormatter(
            "%(log_color)s" + log_format + "%(reset)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    log_file = config.get('file', 'logs/hgvcs.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=config.get('backup_count', 5)
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    return logger

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'cv2', 'numpy', 'mediapipe', 'tensorflow', 'whisper',
        'pyautogui', 'PyQt5', 'yaml', 'websockets', 'zeroconf'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Error: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install dependencies: pip install -r requirements.txt")
        return False
    
    return True

def signal_handler(signum, frame):
    """Handle system signals."""
    logger = logging.getLogger('hgvcs')
    logger.info(f"Received signal {signum}, shutting down...")
    
    # Cleanup will be handled by the application
    QApplication.quit()

def main():
    """Main application entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='HGVCS - Hand Gesture & Voice Control System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start with default settings
  python main.py --debug            # Start in debug mode
  python main.py --gesture-only     # Use only gesture control
  python main.py --voice-only       # Use only voice control
  python main.py --no-gui           # Run without GUI (headless)
        """
    )
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--gesture-only', action='store_true',
                        help='Use only gesture control')
    parser.add_argument('--voice-only', action='store_true',
                        help='Use only voice control')
    parser.add_argument('--no-gui', action='store_true',
                        help='Run without GUI (headless mode)')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run calibration on startup')
    parser.add_argument('--train-gestures', action='store_true',
                        help='Train gesture model and exit')
    parser.add_argument('--collect-data', action='store_true',
                        help='Collect gesture training data')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Load configuration
    import yaml
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.debug:
        config['system']['debug_mode'] = True
        config['logging']['level'] = 'DEBUG'
    
    if args.gesture_only:
        config['fusion']['mode'] = 'gesture_only'
    
    if args.voice_only:
        config['fusion']['mode'] = 'voice_only'
    
    # Setup logging
    logger = setup_logging(config['logging'])
    logger.info("=" * 60)
    logger.info("HGVCS - Hand Gesture & Voice Control System")
    logger.info(f"Version: {config['system']['version']}")
    logger.info("=" * 60)
    
    # Create necessary directories
    for dir_path in ['data', 'logs', 'models', 'certs']:
        os.makedirs(dir_path, exist_ok=True)
    
    # Handle special modes
    if args.train_gestures:
        logger.info("Training gesture model...")
        from scripts.train_gesture_model import train_model
        train_model("data/training/gestures", "models/gesture_model.h5")
        return
    
    if args.collect_data:
        logger.info("Starting gesture data collection...")
        from scripts.collect_gesture_data import collect_data
        collect_data()
        return
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Import Qt and start application
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QIcon
    
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName(config['system']['app_name'])
    app.setApplicationVersion(config['system']['version'])
    
    # Set application style
    if config['ui'].get('theme') == 'dark':
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    gesture_controller = None
    voice_controller   = None
    network_manager    = None
    system_controller  = None
    profile_manager    = None
    macro_engine       = None
    
    try:
        # Initialize core components
        logger.info("Initializing core components...")
        
        from src.core.config import ConfigManager
        from src.core.event_bus import EventBus
        from src.core.state_manager import StateManager
        
        config_manager = ConfigManager(config)
        event_bus = EventBus()
        state_manager = StateManager()
        
        # Initialize gesture recognition
        if config['gesture']['enabled'] and config['fusion']['mode'] != 'voice_only':
            logger.info("Initializing gesture recognition...")
            from src.gesture.gesture_controller import GestureController
            from src.control.system_controller import SystemController
            from src.gesture.macro_engine import MacroEngine
            from src.core.user_profiles import ProfileManager
            system_controller  = SystemController()
            profile_manager    = ProfileManager()
            macro_engine       = MacroEngine()
            gesture_controller = GestureController(config['gesture'], event_bus)
            gesture_controller.set_system_controller(system_controller)
            gesture_controller.set_macro_engine(macro_engine)
            gesture_controller.set_profile_manager(profile_manager)
            logger.info(f"Macro engine: {len(macro_engine.all_macros())} macros loaded")
            logger.info(f"Profile: '{profile_manager.active().name}'")

        # Initialize voice recognition
        if config['voice']['enabled'] and config['fusion']['mode'] != 'gesture_only':
            logger.info("Initializing voice recognition...")
            from src.voice.voice_controller import VoiceController
            voice_controller = VoiceController(config['voice'], event_bus)

        # Initialize network module
        if config['network']['enabled']:
            logger.info("Initializing network module...")
            from src.network.network_manager import NetworkManager
            net_cfg = dict(config['network'])
            net_cfg['port'] = 9876
            network_manager = NetworkManager(net_cfg, event_bus)
            if gesture_controller:
                gesture_controller.set_network_manager(network_manager)

        # Initialize input fusion
        from src.fusion.input_fusion import InputFusionEngine
        fusion_engine = InputFusionEngine(config['fusion'], event_bus)

        # system_controller already initialised above with gesture recognition
        if system_controller is None:
            from src.control.system_controller import SystemController
            system_controller = SystemController()
        
        # Create main window or run headless
        if args.no_gui:
            logger.info("Running in headless mode...")
            # Headless mode - just run the controllers
            if gesture_controller:
                gesture_controller.start()
            if voice_controller:
                voice_controller.start()
            if network_manager:
                network_manager.start()
            
            # Keep running
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            # GUI mode
            logger.info("Starting GUI...")
            from src.ui.main_window import MainWindow
            
            window = MainWindow(
                config=config,
                gesture_controller=gesture_controller,
                voice_controller=voice_controller,
                network_manager=network_manager,
                fusion_engine=fusion_engine,
                system_controller=system_controller,
                profile_manager=profile_manager,
                macro_engine=macro_engine,
            )
            
            # Run calibration if needed
            if args.calibrate or (config['calibration']['required_on_first_run'] and 
                                   not os.path.exists(config['calibration']['calibration_file'])):
                logger.info("Running user calibration...")
                window.run_calibration()
            
            window.show()
            
            # Start controllers
            if gesture_controller:
                gesture_controller.start()
            if voice_controller:
                voice_controller.start()
            if network_manager:
                network_manager.start()
            
            logger.info("Application started successfully!")
            
            # Run application
            sys.exit(app.exec_())
    
    except Exception as e:
        logger.exception("Fatal error during startup")
        print(f"\nError: {e}")
        print("\nFor help, see documentation at: docs/README.md")
        sys.exit(1)
    
    finally:
        # Cleanup
        logger.info("Shutting down...")
        if gesture_controller:
            gesture_controller.stop()
        if voice_controller:
            voice_controller.stop()
        if network_manager:
            network_manager.stop()

if __name__ == "__main__":
    main()
