# comfyui-bhtools/nodes/CinematicSceneDirectorBHTools.py

import random
import os

class CinematicSceneDirector:
    """
    Advanced Cinematic Scene Director Node for ComfyUI
    
    Creates sophisticated cinematic prompts by cycling through master prompts
    and applying comprehensive cinematic parameters including composition,
    lighting, atmosphere, and post-production effects.
    
    Features:
    - Master prompts system with auto-incrementing
    - Comprehensive cinematic controls
    - Weighted parameter system
    - Preset overrides for quick styling
    - Professional film terminology
    """
    
    def __init__(self):
        self.current_index = 0
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "master_prompts": ("STRING", {
                    "default": "A dramatic scene in golden hour lighting \nA mysterious figure walking through fog \nA vibrant marketplace bustling with activity \nA serene mountain landscape at dawn \nAn intense character portrait with dramatic shadows", 
                    "multiline": True,
                    "tooltip": "Enter multiple prompts separated by line breaks. The node will cycle through them automatically"
                }),
                "prompt_index": ("INT", {
                    "default": 0, 
                    "min": -1, 
                    "max": 999, 
                    "step": 1,
                    "tooltip": "Current prompt index (auto-increments). Set to -1 for random selection."
                }),
                "auto_increment": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically increment to next prompt on each run"
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show debug information about prompt construction"
                }),
            },
            "optional": {
                # Group 1: Quick Presets
                "preset_override": ([
                    "None",
                    # Cinematic Styles
                    "Cinematic Drama", "Epic Cinematic", "Intimate Drama", "Blockbuster Action",
                    # Lighting Moods  
                    "Golden Hour Magic", "Blue Hour Mystery", "Noir Shadows", "High Key Bright",
                    # Genre Specific
                    "Sci-Fi Futuristic", "Fantasy Epic", "Horror Atmospheric", "Romance Dreamy",
                    "Western Gritty", "Cyberpunk Neon", "Period Historical", "Documentary Real",
                    # Visual Styles
                    "Vintage Film", "Modern Digital", "Artistic Abstract", "Hyperrealistic",
                    "Minimalist Clean", "Maximalist Rich", "Retro Nostalgic", "Avant Garde",
                    # Environmental
                    "Urban Cityscape", "Natural Landscape", "Interior Intimate", "Cosmic Space",
                    "Underwater Depths", "Desert Vastness", "Forest Mystical", "Mountain Majestic"
                ], {"default": "None"}),

                # Group 2: Shot Composition
                "shot_type": ([
                    "None",
                    "Extreme Wide Shot", "Wide Shot", "Full Shot", "Medium Full Shot",
                    "Medium Shot", "Medium Close-Up", "Close-Up", "Extreme Close-Up",
                    "Two-Shot", "Over-the-Shoulder", "Point of View", "Insert Shot",
                    "Cutaway", "Establishing Shot", "Master Shot"
                ], {"default": "None"}),
                "shot_type_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                "camera_angle": ([
                    "None",
                    "Eye Level", "High Angle", "Low Angle", "Bird's Eye View", "Worm's Eye View",
                    "Dutch Angle", "Overhead Shot", "Aerial View", "Ground Level",
                    "Shoulder Level", "Hip Level", "Knee Level"
                ], {"default": "None"}),
                "camera_angle_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                "camera_movement": ([
                    "None",
                    "Static Shot", "Pan Left", "Pan Right", "Tilt Up", "Tilt Down",
                    "Dolly In", "Dolly Out", "Truck Left", "Truck Right", "Pedestal Up", "Pedestal Down",
                    "Zoom In", "Zoom Out", "Rack Focus", "Handheld", "Steadicam", "Gimbal",
                    "Crane Up", "Crane Down", "Jib Arm", "Slider", "Circular Dolly", "Parallax"
                ], {"default": "None"}),
                "camera_movement_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                # Group 3: Lens & Focus
                "camera_lens": ([
                    "None",
                    "Ultra Wide 14mm", "Wide Angle 24mm", "Standard 35mm", "Portrait 50mm", 
                    "Short Telephoto 85mm", "Telephoto 135mm", "Super Telephoto 200mm",
                    "Fisheye", "Macro", "Tilt-Shift", "Anamorphic", "Vintage Lens", "Prime Lens", "Zoom Lens"
                ], {"default": "None"}),
                "camera_lens_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                "depth_of_field": ([
                    "None",
                    "Shallow DOF", "Deep DOF", "Rack Focus", "Split Focus", "Hyperfocal",
                    "Bokeh Heavy", "Tilt-Shift Miniature", "Focus Pulling", "Soft Focus",
                    "Crisp Focus", "Selective Focus", "Zone Focus"
                ], {"default": "None"}),
                "depth_of_field_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                # Group 4: Lighting Setup
                "lighting_setup": ([
                    "None",
                    "Three-Point Lighting", "Key Light Only", "Rembrandt Lighting", "Butterfly Lighting",
                    "Split Lighting", "Loop Lighting", "Broad Lighting", "Short Lighting",
                    "Rim Lighting", "Backlighting", "Side Lighting", "Top Lighting", "Bottom Lighting",
                    "Practical Lighting", "Motivated Lighting", "Available Light"
                ], {"default": "None"}),
                "lighting_setup_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                "lighting_quality": ([
                    "None",
                    "Hard Light", "Soft Light", "Diffused Light", "Direct Light", "Bounced Light",
                    "Filtered Light", "Natural Light", "Artificial Light", "Mixed Lighting",
                    "High Contrast", "Low Contrast", "Even Lighting", "Dramatic Lighting",
                    "Subtle Lighting", "Harsh Lighting", "Gentle Lighting"
                ], {"default": "None"}),
                "lighting_quality_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                "lighting_color": ([
                    "None",
                    "Warm 3200K", "Neutral 5600K", "Cool 7000K", "Daylight 5500K", "Tungsten 3200K",
                    "LED 4000K", "Fluorescent 4100K", "Candlelight 1900K", "Sunset 2000K",
                    "Blue Hour 15000K", "Overcast 6500K", "Golden Hour 2500K",
                    "Neon Colors", "Colored Gels", "Practical Colors"
                ], {"default": "None"}),
                "lighting_color_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                # Group 5: Time & Environment
                "time_of_day": ([
                    "None",
                    "Pre-Dawn", "Dawn", "Early Morning", "Mid Morning", "Late Morning",
                    "Noon", "Early Afternoon", "Mid Afternoon", "Late Afternoon",
                    "Dusk", "Twilight", "Blue Hour", "Golden Hour", "Night",
                    "Late Night", "Midnight", "Pre-Dawn Hours"
                ], {"default": "None"}),
                "time_of_day_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                "season": ([
                    "None",
                    "Spring", "Summer", "Autumn", "Winter",
                    "Early Spring", "Late Spring", "Mid Summer", "Late Summer",
                    "Early Fall", "Late Fall", "Early Winter", "Late Winter"
                ], {"default": "None"}),
                "season_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                "environment_type": ([
                    "None",
                    "Interior", "Exterior", "Studio", "Location", "Practical Location",
                    "Urban", "Suburban", "Rural", "Wilderness", "Industrial", "Residential",
                    "Commercial", "Institutional", "Historical", "Modern", "Futuristic"
                ], {"default": "None"}),
                "environment_type_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                # Group 6: Atmospheric Conditions (Separated from Time)
                "weather_atmosphere": ([
                    "None",
                    "Clear", "Partly Cloudy", "Overcast", "Stormy", "Rainy", "Drizzling",
                    "Heavy Rain", "Thunderstorm", "Snowing", "Light Snow", "Heavy Snow", "Blizzard",
                    "Foggy", "Misty", "Hazy", "Dusty", "Windy", "Calm", "Humid", "Dry"
                ], {"default": "None"}),
                "weather_atmosphere_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                "atmospheric_effects": ([
                    "None",
                    "Volumetric Light", "God Rays", "Lens Flare", "Light Shafts", "Dust Particles",
                    "Smoke", "Steam", "Mist", "Fog Bank", "Heat Shimmer", "Light Pollution",
                    "Atmospheric Haze", "Particle Effects", "Floating Debris", "Pollen", "Ash"
                ], {"default": "None"}),
                "atmospheric_effects_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                # Group 7: Visual Style & Mood
                "visual_style": ([
                    "None",
                    "Photorealistic", "Stylized", "Artistic", "Abstract", "Surreal", "Impressionistic",
                    "Expressionistic", "Minimalist", "Maximalist", "Geometric", "Organic",
                    "Textural", "Smooth", "Rough", "Polished", "Raw", "Refined", "Gritty"
                ], {"default": "None"}),
                "visual_style_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                "mood_emotion": ([
                    "None",
                    "Joyful", "Melancholic", "Mysterious", "Dramatic", "Peaceful", "Tense",
                    "Romantic", "Threatening", "Whimsical", "Serious", "Playful", "Somber",
                    "Energetic", "Calm", "Chaotic", "Ordered", "Intimate", "Epic",
                    "Nostalgic", "Futuristic", "Timeless", "Urgent", "Contemplative"
                ], {"default": "None"}),
                "mood_emotion_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                "genre_influence": ([
                    "None",
                    "Film Noir", "Science Fiction", "Fantasy", "Horror", "Thriller", "Drama",
                    "Comedy", "Romance", "Action", "Adventure", "Western", "Crime",
                    "Mystery", "Documentary", "Art House", "Experimental", "Period Piece",
                    "Biopic", "Musical", "War Film", "Disaster", "Superhero"
                ], {"default": "None"}),
                "genre_influence_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                # Group 8: Color & Aesthetics
                "color_grading": ([
                    "None",
                    "Natural", "Warm Tones", "Cool Tones", "Desaturated", "High Saturation",
                    "Monochromatic", "Complementary", "Analogous", "Triadic", "Split Complementary",
                    "Teal and Orange", "Vintage", "Modern", "Faded", "Vibrant",
                    "Muted Palette", "Earth Tones", "Neon Palette", "Pastel Palette", "Dark Palette"
                ], {"default": "None"}),
                "color_grading_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                "film_emulation": ([
                    "None",
                    "Kodak Vision3", "Fuji Eterna", "Arri Alexa", "Red Dragon", "Digital",
                    "35mm Film", "16mm Film", "Super 8", "Vintage Film", "Modern Digital",
                    "Film Grain Heavy", "Film Grain Light", "No Grain", "Digital Noise",
                    "Clean Digital", "Organic Texture"
                ], {"default": "None"}),
                "film_emulation_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                # Group 9: Technical Quality
                "image_quality": ([
                    "None",
                    "Standard Definition", "High Definition", "Ultra HD 4K", "8K Resolution",
                    "Cinematic 4K", "IMAX Quality", "Web Optimized", "Print Quality",
                    "Broadcast Quality", "Cinema Quality", "Ultra High Quality",
                    "Maximum Detail", "Sharp", "Soft", "Crisp", "Professional Grade"
                ], {"default": "None"}),
                "image_quality_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                "aspect_ratio": ([
                    "None",
                    "16:9 Widescreen", "21:9 Ultrawide", "2.35:1 Anamorphic", "1.85:1 Academy",
                    "4:3 Classic", "1:1 Square", "9:16 Vertical", "3:2 Photography",
                    "5:4 Large Format", "2.39:1 Cinemascope", "1.33:1 Academy Classic",
                    "2.76:1 Ultra Panavision", "Custom Ratio"
                ], {"default": "None"}),
                "aspect_ratio_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                # Group 10: Post-Production Effects
                "lens_effects": ([
                    "None",
                    "Lens Flare", "Chromatic Aberration", "Vignetting", "Barrel Distortion",
                    "Pincushion Distortion", "Bokeh Highlights", "Lens Ghosting", "Glare",
                    "Diffraction Spikes", "Anamorphic Flares", "Vintage Lens Character",
                    "Clean Lens", "Dirty Lens", "Water Drops", "Lens Breathing"
                ], {"default": "None"}),
                "lens_effects_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                "post_processing": ([
                    "None",
                    "HDR", "Tone Mapping", "Color Correction", "Color Grading", "Contrast Enhancement",
                    "Saturation Boost", "Desaturation", "Sharpening", "Noise Reduction",
                    "Film Emulation", "Digital Enhancement", "Vintage Processing", "Modern Processing",
                    "Artistic Filter", "Realistic Processing", "Stylized Processing"
                ], {"default": "None"}),
                "post_processing_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                # Group 11: Advanced Cinematic Techniques
                "composition_rule": ([
                    "None",
                    "Rule of Thirds", "Golden Ratio", "Center Composition", "Symmetrical",
                    "Asymmetrical", "Leading Lines", "Framing", "Depth Layering",
                    "Foreground Focus", "Background Focus", "Negative Space", "Filling Frame",
                    "Diagonal Composition", "Triangular Composition", "Circular Composition"
                ], {"default": "None"}),
                "composition_rule_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),

                "cinematic_technique": ([
                    "None",
                    "Montage", "Long Take", "Quick Cuts", "Match Cut", "Jump Cut", "Cross Cut",
                    "Fade In", "Fade Out", "Dissolve", "Wipe", "Iris", "Split Screen",
                    "Freeze Frame", "Slow Motion", "Time Lapse", "Reverse Motion",
                    "Seamless Transition", "Hard Cut"
                ], {"default": "None"}),
                "cinematic_technique_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1})
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("cinematic_prompt", "next_index", "debug_info")
    FUNCTION = "generate_prompt_string"
    CATEGORY = "BH Nodes 🎬"

    def parse_master_prompts(self, master_prompts_text):
        """Parse the master prompts from multiline text, filtering out empty lines."""
        if not master_prompts_text or not master_prompts_text.strip():
            return ["A cinematic scene."]
        
        prompts = [line.strip() for line in master_prompts_text.split('\n') if line.strip()]
        
        if not prompts:
            return ["A cinematic scene."]
            
        return prompts

    def get_current_prompt(self, master_prompts_list, prompt_index, auto_increment):
        """Get the current prompt based on index and increment logic."""
        if not master_prompts_list:
            return "A cinematic scene.", 0
            
        total_prompts = len(master_prompts_list)
        
        if prompt_index == -1:
            selected_index = random.randint(0, total_prompts - 1)
            next_index = -1
            return master_prompts_list[selected_index], next_index
        
        current_index = prompt_index % total_prompts
        current_prompt = master_prompts_list[current_index]
        
        if auto_increment:
            next_index = (current_index + 1) % total_prompts
        else:
            next_index = current_index
            
        return current_prompt, next_index

    def weighted_format(self, value, weight):
        """Format a value with its weight for prompt inclusion."""
        if value == "None" or not value.strip():
            return None
        value = value.lower()
        rounded_weight = round(weight, 1)
        if rounded_weight == 1.0:
            return value
        else:
            return f"({value}:{rounded_weight})"

    def format_setting(self, value, weight, prefix="", suffix=""):
        """Format a setting with optional prefix and suffix."""
        formatted = self.weighted_format(value, weight)
        if formatted is None:
            return None
        return f"{prefix}{formatted}{suffix}"

    def generate_prompt_string(self, **kwargs):
        """
        Main function to generate the cinematic prompt string.
        Combines master prompts with comprehensive cinematic parameters.
        """
        
        # Extract required parameters
        master_prompts = kwargs.get('master_prompts', '')
        prompt_index = kwargs.get('prompt_index', 0)
        auto_increment = kwargs.get('auto_increment', True)
        enable_debug = kwargs.get('enable_debug', False)
        
        # Parse and get current master prompt
        master_prompts_list = self.parse_master_prompts(master_prompts)
        current_master_prompt, next_index = self.get_current_prompt(
            master_prompts_list, prompt_index, auto_increment
        )
        
        debug_info = []
        if enable_debug:
            debug_info.append(f"Master prompts count: {len(master_prompts_list)}")
            debug_info.append(f"Current index: {prompt_index}")
            debug_info.append(f"Selected prompt: {current_master_prompt}")
            debug_info.append(f"Next index: {next_index}")
        
        # Enhanced preset definitions
        PRESETS = {
            "Cinematic Drama": "visually striking cinematic scene with deep shadows, rich contrast, and dynamic composition",
            "Epic Cinematic": "grand sweeping cinematic vista with dramatic scale and breathtaking composition",
            "Intimate Drama": "close personal cinematic moment with subtle lighting and emotional depth",
            "Blockbuster Action": "high-energy cinematic action scene with dynamic movement and bold lighting",
            "Golden Hour Magic": "warm golden hour cinematic scene with soft directional lighting and magical atmosphere",
            "Blue Hour Mystery": "mysterious blue hour cinematic scene with cool tones and atmospheric depth",
            "Noir Shadows": "film noir cinematic scene with stark contrasts and dramatic shadow play",
            "High Key Bright": "bright high-key cinematic scene with even lighting and minimal shadows",
            "Sci-Fi Futuristic": "futuristic sci-fi cinematic scene with advanced technology and sleek aesthetics",
            "Fantasy Epic": "epic fantasy cinematic scene with magical elements and otherworldly beauty",
            "Horror Atmospheric": "atmospheric horror cinematic scene with eerie lighting and unsettling mood",
            "Romance Dreamy": "romantic dreamy cinematic scene with soft lighting and warm emotional tones",
            "Western Gritty": "gritty western cinematic scene with dusty atmosphere and harsh lighting",
            "Cyberpunk Neon": "cyberpunk cinematic scene with neon lighting and urban futuristic elements",
            "Period Historical": "historically accurate cinematic scene with period-appropriate styling and atmosphere",
            "Documentary Real": "realistic documentary-style cinematic scene with natural lighting and authentic feel",
            "Vintage Film": "vintage film cinematic scene with classic film grain and retro aesthetic",
            "Modern Digital": "modern digital cinematic scene with clean lines and contemporary styling",
            "Artistic Abstract": "artistic abstract cinematic scene with creative composition and stylized elements",
            "Hyperrealistic": "hyperrealistic cinematic scene with incredible detail and lifelike quality",
            "Minimalist Clean": "minimalist cinematic scene with clean composition and subtle elements",
            "Maximalist Rich": "rich maximalist cinematic scene with complex details and layered composition",
            "Retro Nostalgic": "nostalgic retro cinematic scene with vintage styling and warm tones",
            "Avant Garde": "avant-garde cinematic scene with experimental composition and bold artistic choices",
            "Urban Cityscape": "urban cityscape cinematic scene with metropolitan atmosphere and architectural elements",
            "Natural Landscape": "natural landscape cinematic scene with organic beauty and environmental depth",
            "Interior Intimate": "intimate interior cinematic scene with controlled lighting and personal atmosphere",
            "Cosmic Space": "cosmic space cinematic scene with stellar elements and infinite depth",
            "Underwater Depths": "underwater cinematic scene with aquatic lighting and fluid movement",
            "Desert Vastness": "vast desert cinematic scene with expansive horizons and harsh beauty",
            "Forest Mystical": "mystical forest cinematic scene with dappled light and natural mystery",
            "Mountain Majestic": "majestic mountain cinematic scene with dramatic elevation and natural grandeur"
        }
        
        # Collect all prompt components
        components = []
        
        # Apply preset override
        preset_override = kwargs.get('preset_override', 'None')
        if preset_override != "None" and preset_override in PRESETS:
            components.append(PRESETS[preset_override])
            if enable_debug:
                debug_info.append(f"Applied preset: {preset_override}")
        
        # Process all parameters systematically
        param_groups = {
            "Shot Composition": [
                ('shot_type', 'shot_type_weight', '', ' shot'),
                ('camera_angle', 'camera_angle_weight', 'from ', ' angle'),
                ('camera_movement', 'camera_movement_weight', 'with ', ' camera movement'),
            ],
            "Lens & Focus": [
                ('camera_lens', 'camera_lens_weight', 'shot with ', ' lens'),
                ('depth_of_field', 'depth_of_field_weight', 'featuring ', ''),
            ],
            "Lighting": [
                ('lighting_setup', 'lighting_setup_weight', 'using ', ''),
                ('lighting_quality', 'lighting_quality_weight', 'with ', ''),
                ('lighting_color', 'lighting_color_weight', 'in ', ' color temperature'),
            ],
            "Time & Environment": [
                ('time_of_day', 'time_of_day_weight', 'during ', ''),
                ('season', 'season_weight', 'in ', ''),
                ('environment_type', 'environment_type_weight', 'in ', ' setting'),
            ],
            "Atmosphere": [
                ('weather_atmosphere', 'weather_atmosphere_weight', 'with ', ' weather'),
                ('atmospheric_effects', 'atmospheric_effects_weight', 'featuring ', ''),
            ],
            "Style & Mood": [
                ('visual_style', 'visual_style_weight', 'in ', ' style'),
                ('mood_emotion', 'mood_emotion_weight', 'with ', ' mood'),
                ('genre_influence', 'genre_influence_weight', 'evoking ', ' genre'),
            ],
            "Color & Aesthetic": [
                ('color_grading', 'color_grading_weight', 'with ', ' color grading'),
                ('film_emulation', 'film_emulation_weight', 'emulating ', ''),
            ],
            "Technical": [
                ('image_quality', 'image_quality_weight', 'rendered at ', ''),
                ('aspect_ratio', 'aspect_ratio_weight', 'in ', ' aspect ratio'),
            ],
            "Post-Production": [
                ('lens_effects', 'lens_effects_weight', 'with ', ''),
                ('post_processing', 'post_processing_weight', 'processed with ', ''),
            ],
            "Advanced Techniques": [
                ('composition_rule', 'composition_rule_weight', 'composed using ', ''),
                ('cinematic_technique', 'cinematic_technique_weight', 'utilizing ', ' technique'),
            ]
        }
        
        # Process each parameter group
        for group_name, params in param_groups.items():
            group_components = []
            for param_name, weight_name, prefix, suffix in params:
                value = kwargs.get(param_name, 'None')
                weight = kwargs.get(weight_name, 1.0)
                formatted = self.format_setting(value, weight, prefix, suffix)
                if formatted:
                    group_components.append(formatted)
            
            if group_components and enable_debug:
                debug_info.append(f"{group_name}: {', '.join(group_components)}")
            
            components.extend(group_components)
        
        # Assemble final prompt
        final_prompt = current_master_prompt.strip()
        
        if components:
            components_text = ", ".join(components)
            if final_prompt:
                final_prompt += ", " + components_text
            else:
                final_prompt = components_text
        
        # Final cleanup
        final_prompt = final_prompt.replace(",,", ",").replace(" ,", ",").replace("  ", " ").strip()
        if final_prompt and not final_prompt.endswith((".", "!", "?")):
            final_prompt += "."
        
        # Prepare debug output
        debug_output = "\n".join(debug_info) if enable_debug else ""
        
        return (final_prompt, next_index, debug_output)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "CinematicSceneDirectorBHTools": CinematicSceneDirector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CinematicSceneDirectorBHTools": "🎬 Cinematic Scene Director | BH Tools",
}