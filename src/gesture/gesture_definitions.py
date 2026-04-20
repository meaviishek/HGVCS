"""
Gesture Definitions Module

This module defines all 25+ gestures supported by HGVCS,
including their recognition criteria and action mappings.
"""

from dataclasses import dataclass
from typing import List, Dict, Callable, Optional
from enum import Enum
import numpy as np

class GestureCategory(Enum):
    """Categories of gestures."""
    SYSTEM_CONTROL = "system_control"
    NAVIGATION = "navigation"
    MEDIA_CONTROL = "media_control"
    FILE_OPERATION = "file_operation"
    NETWORK_SHARING = "network_sharing"
    ACCESSIBILITY = "accessibility"

@dataclass
class GestureDefinition:
    """Definition of a hand gesture."""
    name: str
    display_name: str
    category: GestureCategory
    description: str
    confidence_threshold: float
    action: str
    action_description: str
    
    # Recognition criteria
    finger_states: Dict[str, str]  # thumb, index, middle, ring, pinky: 'extended', 'folded', 'any'
    relative_positions: Optional[List[Dict]] = None  # Relative positions of landmarks
    motion_required: bool = False  # Whether motion is required (for swipes)
    motion_direction: Optional[str] = None  # 'left', 'right', 'up', 'down', 'circular'
    two_hands: bool = False  # Whether gesture requires two hands
    
    # Visual representation (ASCII art)
    visual_representation: str = ""

# =============================================================================
# GESTURE DEFINITIONS
# =============================================================================

GESTURE_DEFINITIONS = {
    # ==========================================================================
    # SYSTEM CONTROL GESTURES
    # ==========================================================================
    
    "open_palm": GestureDefinition(
        name="open_palm",
        display_name="Open Palm",
        category=GestureCategory.SYSTEM_CONTROL,
        description="All fingers extended, palm facing camera",
        confidence_threshold=0.85,
        action="pause",
        action_description="Pause current operation",
        finger_states={
            "thumb": "extended",
            "index": "extended",
            "middle": "extended",
            "ring": "extended",
            "pinky": "extended"
        },
        visual_representation="""
    _ _ _ _
   |       |
   | | | | |
   | | | | |
   |_|_|_|_|
        """
    ),
    
    "closed_fist": GestureDefinition(
        name="closed_fist",
        display_name="Closed Fist",
        category=GestureCategory.SYSTEM_CONTROL,
        description="All fingers curled into palm",
        confidence_threshold=0.80,
        action="confirm",
        action_description="Confirm/Select action",
        finger_states={
            "thumb": "folded",
            "index": "folded",
            "middle": "folded",
            "ring": "folded",
            "pinky": "folded"
        },
        visual_representation="""
    _______
   |       |
   | | | | |
   |_|_|_|_|
        """
    ),
    
    "thumbs_up": GestureDefinition(
        name="thumbs_up",
        display_name="Thumbs Up",
        category=GestureCategory.SYSTEM_CONTROL,
        description="Thumb extended upward, other fingers folded",
        confidence_threshold=0.85,
        action="accept",
        action_description="Accept/Yes/Continue",
        finger_states={
            "thumb": "extended",
            "index": "folded",
            "middle": "folded",
            "ring": "folded",
            "pinky": "folded"
        },
        relative_positions=[
            {"thumb_tip": "above", "reference": "wrist"}
        ],
        visual_representation="""
    _ _ _ _
   | | | | |
   | | | | |
   | | | | |
   _|_|_|_|_
        """
    ),
    
    "thumbs_down": GestureDefinition(
        name="thumbs_down",
        display_name="Thumbs Down",
        category=GestureCategory.SYSTEM_CONTROL,
        description="Thumb extended downward, other fingers folded",
        confidence_threshold=0.85,
        action="reject",
        action_description="Reject/No/Cancel",
        finger_states={
            "thumb": "extended",
            "index": "folded",
            "middle": "folded",
            "ring": "folded",
            "pinky": "folded"
        },
        relative_positions=[
            {"thumb_tip": "below", "reference": "wrist"}
        ],
        visual_representation="""
    _ _ _ _
   | | | | |
   | | | | |
   |_|_|_|_|
      | |
        """
    ),
    
    "pointing": GestureDefinition(
        name="pointing",
        display_name="Pointing",
        category=GestureCategory.SYSTEM_CONTROL,
        description="Index finger extended, others folded",
        confidence_threshold=0.75,
        action="cursor_mode",
        action_description="Enter cursor control mode",
        finger_states={
            "thumb": "any",
            "index": "extended",
            "middle": "folded",
            "ring": "folded",
            "pinky": "folded"
        },
        visual_representation="""
    _ _ _ _
   | | | | |
   | | | | |
   | | |_|_|
   |_|_|_|
        """
    ),
    
    "peace_sign": GestureDefinition(
        name="peace_sign",
        display_name="Peace Sign",
        category=GestureCategory.SYSTEM_CONTROL,
        description="Index and middle fingers extended in V shape",
        confidence_threshold=0.80,
        action="screenshot",
        action_description="Take screenshot",
        finger_states={
            "thumb": "folded",
            "index": "extended",
            "middle": "extended",
            "ring": "folded",
            "pinky": "folded"
        },
        visual_representation="""
    _ _ _ _
   | | | | |
   | | | | |
   | |_|_|_|
   |_|_|_|
        """
    ),
    
    "ok_sign": GestureDefinition(
        name="ok_sign",
        display_name="OK Sign",
        category=GestureCategory.SYSTEM_CONTROL,
        description="Thumb and index finger form circle, others extended",
        confidence_threshold=0.85,
        action="open_settings",
        action_description="Open settings menu",
        finger_states={
            "thumb": "extended",
            "index": "extended",
            "middle": "extended",
            "ring": "extended",
            "pinky": "extended"
        },
        relative_positions=[
            {"thumb_tip": "near", "reference": "index_tip", "threshold": 0.05}
        ],
        visual_representation="""
     _ _ _
    /     \
   |   _   |
    \\__|__/
        """
    ),
    
    "rock_on": GestureDefinition(
        name="rock_on",
        display_name="Rock On",
        category=GestureCategory.SYSTEM_CONTROL,
        description="Index and pinky extended, others folded",
        confidence_threshold=0.80,
        action="toggle_media",
        action_description="Toggle play/pause",
        finger_states={
            "thumb": "folded",
            "index": "extended",
            "middle": "folded",
            "ring": "folded",
            "pinky": "extended"
        },
        visual_representation="""
    _ _ _ _
   | | | | |
   |_|_|_|_|
      |   |
      |   |
        """
    ),
    
    "three_fingers": GestureDefinition(
        name="three_fingers",
        display_name="Three Fingers",
        category=GestureCategory.SYSTEM_CONTROL,
        description="Index, middle, and ring fingers extended",
        confidence_threshold=0.75,
        action="volume_up",
        action_description="Increase volume",
        finger_states={
            "thumb": "folded",
            "index": "extended",
            "middle": "extended",
            "ring": "extended",
            "pinky": "folded"
        },
        visual_representation="""
    _ _ _ _
   | | | | |
   | | | | |
   | | | |_|_
   |_|_|_|
        """
    ),
    
    "four_fingers": GestureDefinition(
        name="four_fingers",
        display_name="Four Fingers",
        category=GestureCategory.SYSTEM_CONTROL,
        description="All fingers except thumb extended",
        confidence_threshold=0.75,
        action="volume_down",
        action_description="Decrease volume",
        finger_states={
            "thumb": "folded",
            "index": "extended",
            "middle": "extended",
            "ring": "extended",
            "pinky": "extended"
        },
        visual_representation="""
    _ _ _ _
   | | | | |
   | | | | |
   | | | | |
   |_|_|_|_|
        """
    ),
    
    "cross_fingers": GestureDefinition(
        name="cross_fingers",
        display_name="Cross Fingers",
        category=GestureCategory.SYSTEM_CONTROL,
        description="Index and middle fingers crossed",
        confidence_threshold=0.85,
        action="lock_screen",
        action_description="Lock screen",
        finger_states={
            "thumb": "any",
            "index": "extended",
            "middle": "extended",
            "ring": "any",
            "pinky": "any"
        },
        relative_positions=[
            {"index_tip": "near", "reference": "middle_tip", "threshold": 0.03}
        ],
        visual_representation="""
    _ _ _ _
   | | | | |
   | |\ /| |
   | | X | |
   |_|/_\\_|_|
        """
    ),
    
    # ==========================================================================
    # NAVIGATION GESTURES
    # ==========================================================================
    
    "swipe_left": GestureDefinition(
        name="swipe_left",
        display_name="Swipe Left",
        category=GestureCategory.NAVIGATION,
        description="Open palm moving left across screen",
        confidence_threshold=0.70,
        action="previous_workspace",
        action_description="Switch to previous workspace",
        finger_states={
            "thumb": "extended",
            "index": "extended",
            "middle": "extended",
            "ring": "extended",
            "pinky": "extended"
        },
        motion_required=True,
        motion_direction="left",
        visual_representation="""
    ← ← ← ←
        """
    ),
    
    "swipe_right": GestureDefinition(
        name="swipe_right",
        display_name="Swipe Right",
        category=GestureCategory.NAVIGATION,
        description="Open palm moving right across screen",
        confidence_threshold=0.70,
        action="next_workspace",
        action_description="Switch to next workspace",
        finger_states={
            "thumb": "extended",
            "index": "extended",
            "middle": "extended",
            "ring": "extended",
            "pinky": "extended"
        },
        motion_required=True,
        motion_direction="right",
        visual_representation="""
    → → → →
        """
    ),
    
    "swipe_up": GestureDefinition(
        name="swipe_up",
        display_name="Swipe Up",
        category=GestureCategory.NAVIGATION,
        description="Open palm moving up across screen",
        confidence_threshold=0.70,
        action="scroll_up",
        action_description="Scroll up",
        finger_states={
            "thumb": "extended",
            "index": "extended",
            "middle": "extended",
            "ring": "extended",
            "pinky": "extended"
        },
        motion_required=True,
        motion_direction="up",
        visual_representation="""
      ↑
      ↑
      ↑
      ↑
        """
    ),
    
    "swipe_down": GestureDefinition(
        name="swipe_down",
        display_name="Swipe Down",
        category=GestureCategory.NAVIGATION,
        description="Open palm moving down across screen",
        confidence_threshold=0.70,
        action="scroll_down",
        action_description="Scroll down",
        finger_states={
            "thumb": "extended",
            "index": "extended",
            "middle": "extended",
            "ring": "extended",
            "pinky": "extended"
        },
        motion_required=True,
        motion_direction="down",
        visual_representation="""
      ↓
      ↓
      ↓
      ↓
        """
    ),
    
    "circular_cw": GestureDefinition(
        name="circular_cw",
        display_name="Circular (Clockwise)",
        category=GestureCategory.NAVIGATION,
        description="Circular motion in clockwise direction",
        confidence_threshold=0.75,
        action="refresh",
        action_description="Refresh/Reload",
        finger_states={
            "thumb": "any",
            "index": "extended",
            "middle": "any",
            "ring": "any",
            "pinky": "any"
        },
        motion_required=True,
        motion_direction="circular_cw",
        visual_representation="""
     ↻
        """
    ),
    
    "circular_ccw": GestureDefinition(
        name="circular_ccw",
        display_name="Circular (Counter-Clockwise)",
        category=GestureCategory.NAVIGATION,
        description="Circular motion in counter-clockwise direction",
        confidence_threshold=0.75,
        action="undo",
        action_description="Undo last action",
        finger_states={
            "thumb": "any",
            "index": "extended",
            "middle": "any",
            "ring": "any",
            "pinky": "any"
        },
        motion_required=True,
        motion_direction="circular_ccw",
        visual_representation="""
     ↺
        """
    ),
    
    "pinch_in": GestureDefinition(
        name="pinch_in",
        display_name="Pinch In",
        category=GestureCategory.NAVIGATION,
        description="Thumb and index finger moving closer together",
        confidence_threshold=0.80,
        action="zoom_out",
        action_description="Zoom out",
        finger_states={
            "thumb": "extended",
            "index": "extended",
            "middle": "any",
            "ring": "any",
            "pinky": "any"
        },
        motion_required=True,
        visual_representation="""
      →  O  ←
        """
    ),
    
    "pinch_out": GestureDefinition(
        name="pinch_out",
        display_name="Pinch Out",
        category=GestureCategory.NAVIGATION,
        description="Thumb and index finger moving apart",
        confidence_threshold=0.80,
        action="zoom_in",
        action_description="Zoom in",
        finger_states={
            "thumb": "extended",
            "index": "extended",
            "middle": "any",
            "ring": "any",
            "pinky": "any"
        },
        motion_required=True,
        visual_representation="""
      ←  O  →
        """
    ),
    
    # ==========================================================================
    # APPLICATION GESTURES
    # ==========================================================================
    
    "l_shape": GestureDefinition(
        name="l_shape",
        display_name="L-Shape",
        category=GestureCategory.SYSTEM_CONTROL,
        description="Thumb and index finger form L shape",
        confidence_threshold=0.80,
        action="open_file_manager",
        action_description="Open file manager",
        finger_states={
            "thumb": "extended",
            "index": "extended",
            "middle": "folded",
            "ring": "folded",
            "pinky": "folded"
        },
        relative_positions=[
            {"thumb_tip": "perpendicular", "reference": "index_tip"}
        ],
        visual_representation="""
    |_ _ _
    |
    |
    |
        """
    ),
    
    "phone_sign": GestureDefinition(
        name="phone_sign",
        display_name="Phone Sign",
        category=GestureCategory.SYSTEM_CONTROL,
        description="Thumb and pinky extended, like phone gesture",
        confidence_threshold=0.80,
        action="open_communication",
        action_description="Open communication apps",
        finger_states={
            "thumb": "extended",
            "index": "folded",
            "middle": "folded",
            "ring": "folded",
            "pinky": "extended"
        },
        visual_representation="""
    _ _ _ _
   |_|_|_|_|
            |
            |
        """
    ),
    
    "spider_man": GestureDefinition(
        name="spider_man",
        display_name="Spider-Man",
        category=GestureCategory.SYSTEM_CONTROL,
        description="Index, middle, and thumb spread in web-shooting pose",
        confidence_threshold=0.80,
        action="open_browser",
        action_description="Open web browser",
        finger_states={
            "thumb": "extended",
            "index": "extended",
            "middle": "extended",
            "ring": "folded",
            "pinky": "folded"
        },
        visual_representation="""
    _ _ _
   | | | |
   | | | |
   | | |_|
   |_|_|
        """
    ),
    
    "shaka_sign": GestureDefinition(
        name="shaka_sign",
        display_name="Shaka Sign",
        category=GestureCategory.SYSTEM_CONTROL,
        description="Thumb and pinky extended, hand shaken",
        confidence_threshold=0.75,
        action="open_media_player",
        action_description="Open media player",
        finger_states={
            "thumb": "extended",
            "index": "folded",
            "middle": "folded",
            "ring": "folded",
            "pinky": "extended"
        },
        motion_required=True,
        visual_representation="""
    _ _ _ _
   |_|_|_|_|
            |
            |
   ________|
        """
    ),
    
    "gun_sign": GestureDefinition(
        name="gun_sign",
        display_name="Gun Sign",
        category=GestureCategory.SYSTEM_CONTROL,
        description="Thumb up, index pointing forward",
        confidence_threshold=0.80,
        action="screenshot_region",
        action_description="Screenshot of selected region",
        finger_states={
            "thumb": "extended",
            "index": "extended",
            "middle": "folded",
            "ring": "folded",
            "pinky": "folded"
        },
        relative_positions=[
            {"thumb_tip": "above", "reference": "index_pip"}
        ],
        visual_representation="""
    _
   | | _ _
   |_|_|_|_|
        """
    ),
    
    "high_five": GestureDefinition(
        name="high_five",
        display_name="High Five",
        category=GestureCategory.SYSTEM_CONTROL,
        description="Open palm facing camera, moving toward it",
        confidence_threshold=0.85,
        action="wake_system",
        action_description="Wake system / Hello",
        finger_states={
            "thumb": "extended",
            "index": "extended",
            "middle": "extended",
            "ring": "extended",
            "pinky": "extended"
        },
        relative_positions=[
            {"palm_normal": "toward_camera"}
        ],
        visual_representation="""
    _ _ _ _
   |       |
   | | | | |
   | | | | |
   |_|_|_|_|
        """
    ),
    
    # ==========================================================================
    # FILE SHARING GESTURES
    # ==========================================================================
    
    "grab_move": GestureDefinition(
        name="grab_move",
        display_name="Grab & Move",
        category=GestureCategory.NETWORK_SHARING,
        description="Closed fist moving across screen to select file",
        confidence_threshold=0.75,
        action="select_file",
        action_description="Select file for transfer",
        finger_states={
            "thumb": "folded",
            "index": "folded",
            "middle": "folded",
            "ring": "folded",
            "pinky": "folded"
        },
        motion_required=True,
        visual_representation="""
    [====]→
        """
    ),
    
    "release": GestureDefinition(
        name="release",
        display_name="Release",
        category=GestureCategory.NETWORK_SHARING,
        description="Opening fist to confirm file send",
        confidence_threshold=0.80,
        action="confirm_send",
        action_description="Confirm and send file",
        finger_states={
            "thumb": "extended",
            "index": "extended",
            "middle": "extended",
            "ring": "extended",
            "pinky": "extended"
        },
        visual_representation="""
    _ _ _ _
   |       |
   | | | | |
   | | | | |
   |_|_|_|_|
        """
    ),
    
    "two_hand_pinch": GestureDefinition(
        name="two_hand_pinch",
        display_name="Two-Hand Pinch",
        category=GestureCategory.NETWORK_SHARING,
        description="Both hands pinching to select multiple files",
        confidence_threshold=0.75,
        action="select_multiple",
        action_description="Select multiple files",
        finger_states={
            "thumb": "extended",
            "index": "extended",
            "middle": "any",
            "ring": "any",
            "pinky": "any"
        },
        two_hands=True,
        visual_representation="""
    🤏    🤏
        """
    ),
    
    "wave_goodbye": GestureDefinition(
        name="wave_goodbye",
        display_name="Wave Goodbye",
        category=GestureCategory.NETWORK_SHARING,
        description="Open palm waving side to side",
        confidence_threshold=0.70,
        action="cancel",
        action_description="Cancel operation",
        finger_states={
            "thumb": "extended",
            "index": "extended",
            "middle": "extended",
            "ring": "extended",
            "pinky": "extended"
        },
        motion_required=True,
        visual_representation="""
    👋
        """
    ),
    
    "fist_bump": GestureDefinition(
        name="fist_bump",
        display_name="Fist Bump",
        category=GestureCategory.NETWORK_SHARING,
        description="Two fists coming together to accept file",
        confidence_threshold=0.80,
        action="accept_file",
        action_description="Accept incoming file",
        finger_states={
            "thumb": "folded",
            "index": "folded",
            "middle": "folded",
            "ring": "folded",
            "pinky": "folded"
        },
        two_hands=True,
        visual_representation="""
    ✊  ✊
      💥
        """
    ),
}

def get_gesture_definition(name: str) -> Optional[GestureDefinition]:
    """Get gesture definition by name."""
    return GESTURE_DEFINITIONS.get(name)

def get_gestures_by_category(category: GestureCategory) -> List[GestureDefinition]:
    """Get all gestures in a category."""
    return [g for g in GESTURE_DEFINITIONS.values() if g.category == category]

def get_all_gesture_names() -> List[str]:
    """Get list of all gesture names."""
    return list(GESTURE_DEFINITIONS.keys())

def print_gesture_guide():
    """Print a formatted gesture guide."""
    print("\n" + "="*70)
    print("HGVCS GESTURE REFERENCE GUIDE".center(70))
    print("="*70 + "\n")
    
    for category in GestureCategory:
        gestures = get_gestures_by_category(category)
        if not gestures:
            continue
        
        print(f"\n{category.value.upper().replace('_', ' ')}")
        print("-" * 70)
        
        for gesture in gestures:
            print(f"\n  {gesture.display_name} ({gesture.name})")
            print(f"    Description: {gesture.description}")
            print(f"    Action: {gesture.action_description}")
            print(f"    Confidence: {gesture.confidence_threshold}")
            if gesture.visual_representation:
                for line in gesture.visual_representation.strip().split('\n'):
                    print(f"    {line}")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    print_gesture_guide()
